function formSend() {
  document.getElementById('result-img').setAttribute("src", '../static/images/thumbnail-1.jpeg');
  
  const formData = new FormData();
  const inputImage = document.getElementById('org-img-input').files[0];
  const typeValues = document.getElementsByName('type-radio');

  if(inputImage == null){
    alert("Input a Image");
    return;
  }

  let rendererType = '';
  for(let i = 0; i < typeValues.length; i++){
    if(typeValues[i].checked){
        rendererType = typeValues[i].value;
    }
  }

  formData.append("inputImage", inputImage);
  formData.append("rendererType", rendererType);
    
  fetch(
    '/stylize',
    {
      method: 'POST',
      body: formData,
    }
  )
  .then(response => {
    if (response.status == 200){
      return response
    }
    else{
      throw Error("Error occurs while changing background.")
    }
  })
  .then(response => response.blob())
  .then(blob => URL.createObjectURL(blob))
  .then(imageURL => {
    document.getElementById("result-img").setAttribute("src", imageURL);
  })
  .catch(e =>{
  })
}
  
function setThumbnail(event, id){
  const uploaderId = id + '-input';
  const uploadImagePath = document.getElementById(uploaderId).value;

  if(uploadImagePath.length == 0){
    document.getElementById(id).setAttribute("src", '../static/images/thumbnail-1.jpeg');
  } else{
    const reader = new FileReader();
  
    reader.onload = function(event){
      document.getElementById(id).setAttribute("src", event.target.result);
    };
  
    reader.readAsDataURL(event.target.files[0]);
  }
}