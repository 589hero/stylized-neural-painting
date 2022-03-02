import os
import io
import time
import threading
import argparse
import torch
import torch.optim as optim
from PIL import Image, ImageFile
from queue import Queue, Empty
from flask import Flask, request, render_template, send_file, jsonify
from painter import *

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder='./templates/')

ImageFile.LOAD_TRUNCATED_IMAGES = True
requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

# settings
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
args = parser.parse_args(args=[])
args.renderer = 'oilpaintbrush' # [watercolor, markerpen, oilpaintbrush, rectangle]
args.canvas_color = 'black' # [black, white]
args.canvas_size = 512 # size of the canvas for stroke rendering'
args.keep_aspect_ratio = False # whether to keep input aspect ratio when saving outputs
args.max_m_strokes = 300
args.max_divide = 5 # divide an image up-to max_divide x max_divide patches
args.beta_L1 = 1.0 # weight for L1 loss
args.with_ot_loss = False # set True for imporving the convergence by using optimal transportation loss, but will slow-down the speed
args.beta_ot = 0.1 # weight for optimal transportation loss
args.net_G = 'zou-fusion-net' # renderer architecture
args.renderer_checkpoint_dir = './checkpoints_G_oilpaintbrush' # dir to load the pretrained neu-renderer
args.lr = 0.005 # learning rate for stroke searching
args.output_dir = './output' # dir to save painting results
args.disable_preview = True # disable cv2.imshow, for running remotely without x-display'


def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for request in requests_batch:
                org_img_bytes = request['input'][0]
                request['output'] = run(org_img_bytes)

threading.Thread(target=handle_requests_by_batch).start()


def optimize_x(pt):
    pt._load_checkpoint()
    pt.net_G.eval()

    print('begin drawing...')

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, 128, 128]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, 128, 128]).to(device)

    for pt.m_grid in range(1, pt.max_divide + 1):
        pt.img_batch = utils.img2patches(pt.img_, pt.m_grid, pt.net_G.out_size).to(device)
        pt.G_final_pred_canvas = CANVAS_tmp

        pt.initialize_params()
        pt.x_ctt.requires_grad = True
        pt.x_color.requires_grad = True
        pt.x_alpha.requires_grad = True
        utils.set_requires_grad(pt.net_G, False)

        pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

        pt.step_id = 0
        for pt.anchor_id in range(0, pt.m_strokes_per_block):
            pt.stroke_sampler(pt.anchor_id)
            iters_per_stroke = int(500 / pt.m_strokes_per_block)
            for i in range(iters_per_stroke):
                pt.G_pred_canvas = CANVAS_tmp

                # update x
                pt.optimizer_x.zero_grad()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                pt._forward_pass()
                pt._drawing_step_states()
                pt._backward_x()

                pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
                pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
                pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

                pt.optimizer_x.step()
                pt.step_id += 1

        v = pt._normalize_strokes(pt.x)
        v = pt._shuffle_strokes_and_reshape(v)
        PARAMS = np.concatenate([PARAMS, v], axis=1)
        CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
        CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

    pt._save_stroke_params(PARAMS)
    final_rendered_image = pt._render(PARAMS, save_jpgs=False, save_video=False)

    return final_rendered_image


def save_bytes_to_file(bytes, filename):
    with open(filename, 'wb') as f:
        f.write(bytes)


def run(img_bytes):
    try:
        os.makedirs('images', exist_ok=True)
        save_bytes_to_file(img_bytes, 'images/temp.jpg')
        paths = os.path.join('images','temp.jpg')
        args.img_path = paths
        pt = ProgressivePainter(args=args)
        final_rendered_image = optimize_x(pt)
        formatted = (final_rendered_image * 255 / np.max(final_rendered_image)).astype('uint8')
        img = Image.fromarray(formatted)

        if os.path.isfile(paths):
            os.remove(paths)

        bio = io.BytesIO()
        img.save(bio, "PNG")
        bio.seek(0)

        return bio
    except Exception as e:
        return jsonify({'error': 'Exception occurs while changing background'}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/stylize", methods=["POST"])
def stylize():
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'Too Many Reqeusts'}), 429
    org_img_bytes = request.files['inputImage'].read()
    args.renderer = request.form.get('rendererType', 'oilpaintbrush')
    args.renderer_checkpoint_dir = f'./checkpoints_G_{args.renderer}'

    if args.renderer == 'rectangle':
        args.max_divide = 4

    req = {
        'input': [org_img_bytes]
    }

    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    io = req['output']
    if io == "error":
        return jsonify({'error': 'Server error'}), 500

    return send_file(io, mimetype="image/png")


@app.route("/healthz", methods=["GET"])
def check_health():
    return "healthy", 200


if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
