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
            // pastes the prediction, labels, and active labels into the result div.
            const predictions = data.result;
            const labels = data.labels;
            // activeLabels filters the labels based on the predictions.
            const activeLabels = labels
                // .filter takes two arguments, the label and its index.
                // index === 1 means that the label is active.
                .filter((label, index) => predictions[index] === 1);

            // Gets the result div from the HTML and updates its innerHTML.
            // InnerHTML means that it can contain HTML tags.
            // Strong means that the text is bold.
            // ${variable} is used to insert the value of a variable into a string.
            // In this case it inserts all the active labels joined by a comma.
            // If there are no active labels, it shows 'None'.
            document.getElementById('result').innerHTML = `
            <strong>Detected Labels:</strong> ${activeLabels.join(', ') || 'None'}
        `;
        })
        // catch is used to handle the error if the fetch fails.
        .catch(error => {
            // Shows an error in the result div and in the console.
            console.error('Error:', error);
            document.getElementById('result').innerText = 'Error during prediction.';
        });
};
