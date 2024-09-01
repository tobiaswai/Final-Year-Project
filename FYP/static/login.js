console.log('hello world')

const video = document.getElementById('video-element')
const image = document.getElementById('img-element')
const captureBtn = document.getElementById('capture-btn')
const reload = document.getElementById('reload-btn')

reloadBtn.addEventListener('click', () => {
    window.location.reload()
})

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({video: true})
    .then((stream) => {
        video.srcObject = stream
        const {height, width} = stream.getTracks() [0].getSettings()
        
        captureBtn.addEventListener('click', e=> {
            captureBtn.classList.add('not-visible')
            const track = stream.getVideoTracks() [0]
            const imageCapture = new imageCapture(track)
            console.log(imageCapture)

            imageCapture.takePhoto().then(blob => {
                console.log("took photo:", blob)
                const img = new Image(width, height)
                img.src = URL.createObjectURL(blob)
                image.append(img)

                video.classList.add('not-visible')

                const reader = new FileReader()

                reader.readAsDataURL(blob)
                reader.onload = () => {
                    const base64bata = reader.result
                    console.log(base64bata)

                    const fd = new FormData()
                    fd.append('csrfmiddlewaretoken', csrftoken)
                    fd.append('photo', base64bata)

                    $.ajax({
                        type: 'POST',
                        url: '/classify/',
                        enclype: 'multipart/form-data',
                        data: fd,
                        processData: false,
                        contentType: false,
                        success: (resp) => {
                            console.log(resp)
                            window.location.herf = window.location.origin
                        },
                        error: (err) => {
                            console.log(err)
                        }
                    })
                }
            })
        })
    })
}