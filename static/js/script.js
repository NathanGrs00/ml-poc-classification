document.getElementById('submit-btn').onclick =  function() {
    const text = document.getElementById('user-input').value;
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Prediction: ' + data.result;
    });
}