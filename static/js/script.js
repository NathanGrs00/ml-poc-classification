document.getElementById('submit-btn').onclick = function() {
    const text = document.getElementById('user-input').value;
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    })
        .then(response => response.json())
        .then(data => {
            const predictions = data.result;
            const labels = data.labels;
            const activeLabels = labels
                .filter((label, index) => predictions[index] === 1);

            document.getElementById('result').innerHTML = `
            <strong>Detected Labels:</strong> ${activeLabels.join(', ') || 'None'}
        `;
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('result').innerText = 'Error during prediction.';
        });
};
