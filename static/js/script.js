// When the user submits the input.
document.getElementById('submit-btn').onclick = function() {
    // It takes the value of 'user-input' and stores it in the text variable.
    const text = document.getElementById('user-input').value;
    // This gets the predict endpoint from Flask and sends the user input as a POST request.
    fetch('/predict', {
        method: 'POST',
        // headers to indicate the content type is JSON
        headers: {'Content-Type': 'application/json'},
        // Makes a JSON from the text variable.
        body: JSON.stringify({text: text})
    })
        // .then is used to handle the promise returned by fetch.
        // Converts the response to JSON.
        .then(response => response.json())
        // The data from the response is used to update the HTML content of the result div.
        .then(data => {
            // An array of all the label divs in the HTML.
            const labelDivs = [
                'hate_speech', 'violence', 'directed_hate_speech', 'gender',
                'race', 'religion', 'national_origin', 'sexual_orientation',
                'disability', 'not_hate_speech'
            ];
            // Hide all label divs
            labelDivs.forEach(id => {
                document.getElementById(id).style.display = 'none';
            });

            // Show only active divs and set their text
            data.result.forEach((active, idx) => {
                if (active === 1) {
                    const divId = data.labels[idx];
                    const div = document.getElementById(divId);
                    if (div) {
                        div.style.display = 'block';
                    }
                }
             });

             // Show not_hate_speech if hate_speech is 0
             const hateSpeechIdx = data.labels.indexOf('hate_speech');
             if (hateSpeechIdx !== -1 && data.result[hateSpeechIdx] === 0) {
                document.getElementById('not_hate_speech').style.display = 'block';
             }
        })
        // catch is used to handle the error if the fetch fails.
        .catch(error => {
            // Shows an error in the result div and in the console.
            console.error('Error:', error);
            document.getElementById('result').innerText = 'Error during prediction.';
        });
};
