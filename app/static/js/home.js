const homePageEvent = document.addEventListener("click", predict() {
    console.log("Predicting...");
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({}),
        headers: {
            'Content-Type': 'application/json'
        }
    })
});