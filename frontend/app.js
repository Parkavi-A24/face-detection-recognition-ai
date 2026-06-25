const imageInput = document.getElementById('imageInput');
const detectBtn = document.getElementById('detectBtn');
const resultDiv = document.getElementById('result');
const trainForm = document.getElementById('trainForm');
const trainStatus = document.getElementById('trainStatus');

async function detectFaces() {
    if (!imageInput.files.length) {
        resultDiv.innerHTML = '<div class="message">Please select an image first.</div>';
        return;
    }

    const file = imageInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    resultDiv.innerHTML = '<div class="message">Detecting faces...</div>';

    const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    if (!response.ok) {
        resultDiv.innerHTML = `<div class="message">Error: ${data.error}</div>`;
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        const imageUrl = reader.result;
        renderResults(imageUrl, data.detections);
    };
    reader.readAsDataURL(file);
}

function renderResults(imageUrl, detections) {
    const wrapper = document.createElement('div');
    wrapper.className = 'img-wrapper';

    const img = document.createElement('img');
    img.src = imageUrl;
    wrapper.appendChild(img);

    wrapper.style.maxWidth = '100%';

    img.onload = () => {
        const imgRect = img.getBoundingClientRect();
        const scaleX = img.width / img.naturalWidth;
        const scaleY = img.height / img.naturalHeight;

        detections.forEach(det => {
            const box = document.createElement('div');
            box.className = 'detection-box';
            box.style.left = `${det.x * scaleX}px`;
            box.style.top = `${det.y * scaleY}px`;
            box.style.width = `${det.w * scaleX}px`;
            box.style.height = `${det.h * scaleY}px`;

            const label = document.createElement('div');
            label.className = 'label';
            label.textContent = det.name !== 'Unknown' ? `${det.name} (${Math.round(det.confidence)})` : 'Unknown';
            box.appendChild(label);
            wrapper.appendChild(box);
        });
    };

    resultDiv.innerHTML = '';
    resultDiv.appendChild(wrapper);
}

trainForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const personName = document.getElementById('personName').value.trim();
    const trainImages = document.getElementById('trainImages');

    if (!personName || !trainImages.files.length) {
        trainStatus.innerHTML = '<div class="message">Provide a person name and at least one image.</div>';
        return;
    }

    const formData = new FormData();
    for (const file of trainImages.files) {
        formData.append(personName, file);
    }

    trainStatus.innerHTML = '<div class="message">Training model...</div>';
    const response = await fetch('/api/train', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    if (!response.ok) {
        trainStatus.innerHTML = `<div class="message">Error: ${data.error}</div>`;
        return;
    }

    trainStatus.innerHTML = `<div class="message">${data.message}</div><pre>${JSON.stringify(data.labels, null, 2)}</pre>`;
});

detectBtn.addEventListener('click', detectFaces);
