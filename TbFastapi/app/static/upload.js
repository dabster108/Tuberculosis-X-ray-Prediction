/**
 * TB Chest X-Ray Analyzer
 * JavaScript for handling file uploads and API communication
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const loadingContainer = document.getElementById('loading-container');
    // const lottieContainer = document.querySelector('.lottie-container'); // If needed for specific interactions

    let selectedFile = null;

    console.log('TB Analyzer JS loaded - v2');

    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click()); // Allow click on whole area to trigger input
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    analyzeBtn.addEventListener('click', analyzeImage);
    resetBtn.addEventListener('click', resetUI);

    // Initial state for analyze button (optional, good UX)
    // analyzeBtn.disabled = true; // Disabled until a file is loaded

    // Handle drag over event
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    }

    // Handle drag leave event
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    }

    // Handle drop event
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFiles(e.dataTransfer.files);
        }
    }

    // Handle file selection
    function handleFileSelect(e) {
        console.log('File selected via input');
        if (e.target.files && e.target.files[0]) {
            handleFiles(e.target.files);
        }
    }

    // Process selected files
    function handleFiles(files) {
        const file = files[0];
        console.log('Processing file:', file.name, 'Type:', file.type, 'Size:', file.size);

        // Validate file type
        if (!file.type.match('image.*')) {
            showError('Please select an image file (PNG, JPG, JPEG)');
            return;
        }

        selectedFile = file;
        displayPreview(file);
        // analyzeBtn.disabled = false; // Enable analyze button
    }

    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            console.log('File loaded for preview');
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'block'; // Set display to block first
            // Add .visible class after a short delay to trigger transition
            setTimeout(() => {
                previewContainer.classList.add('visible');
            }, 50); 
        };

        reader.onerror = function(e) {
            console.error('Error reading file:', e);
            showError('Error reading file. Please try a different image.');
            // analyzeBtn.disabled = true;
        };

        reader.readAsDataURL(file);
    }

    // Analyze the uploaded image
    function analyzeImage() {
        console.log('Analyze button clicked');

        if (!selectedFile) {
            showError('Please select an image first');
            return;
        }

        // Show loading screen
        loadingContainer.style.display = 'flex';
        resultContainer.style.display = 'none'; // Hide previous results
        resultContainer.classList.remove('show');


        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        console.log('Sending file to server for analysis:', selectedFile.name);

        // Send to API
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Server response status:', response.status);
            if (!response.ok) {
                return response.json().then(errorData => {
                    // Try to parse errorData.detail if it's a stringified JSON
                    let detailMessage = 'Network response was not ok';
                    if (errorData.detail) {
                        try {
                            const parsedDetail = JSON.parse(errorData.detail);
                            detailMessage = parsedDetail.error || errorData.detail;
                        } catch (e) {
                            detailMessage = errorData.detail;
                        }
                    }
                    throw new Error(detailMessage);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Prediction result:', data);
            displayResult(data);
        })
        .catch(error => {
            console.error('Error during analysis:', error);
            showError(`Analysis error: ${error.message}`);
        })
        .finally(() => {
            // Hide loading screen
            loadingContainer.style.display = 'none';
        });
    }

    // Display the analysis result
    function displayResult(data) {
        let resultClass = '';
        let resultMessage = '';
        
        // Clear previous classes
        resultText.className = 'result-text';


        if (data.prediction === 'Normal') {
            resultClass = 'result-normal'; // This class is for the <h3> inside result-text
            resultMessage = `<h3 class="${resultClass}">✓ No signs of Tuberculosis detected</h3>
                           <p>The X-ray appears normal. Regular check-ups are still recommended.</p>`;
        } else if (data.prediction === 'Tuberculosis') { // Assuming 'Tuberculosis' is the other prediction
            resultClass = 'result-tb'; // This class is for the <h3> inside result-text
            resultMessage = `<h3 class="${resultClass}">⚠ Potential Tuberculosis detected</h3>
                           <p>The analysis indicates possible signs of Tuberculosis. Please consult a medical professional immediately.</p>`;
        } else {
             resultMessage = `<p>Unexpected prediction: ${data.prediction}. Confidence: ${data.confidence !== undefined ? (data.confidence * 100).toFixed(2) + '%' : 'N/A'}</p>`;
        }
        
        // Add confidence if available
        if (data.confidence !== undefined) {
            resultMessage += `<p>Confidence: <strong>${(data.confidence * 100).toFixed(2)}%</strong></p>`;
        }


        resultText.innerHTML = resultMessage;
        resultContainer.style.display = 'block';

        // Add animation effect by adding 'show' class
        // Ensure CSS has .result-container.show { opacity: 1; transform: translateY(0); }
        setTimeout(() => {
            resultContainer.classList.add('show');
        }, 50); // Small delay to ensure transition applies
    }

    // Reset the UI
    function resetUI() {
        console.log('UI reset');
        selectedFile = null;
        previewImage.src = '';
        uploadArea.style.display = 'block';
        previewContainer.style.display = 'none';
        previewContainer.classList.remove('visible'); // Remove visible class
        resultContainer.style.display = 'none';
        resultContainer.classList.remove('show');
        fileInput.value = '';
    }

    // Show error message
    function showError(message) {
        console.error('Error:', message);
        // Replace alert with a more modern notification if available/preferred
        alert(message); // Simple alert for now
    }

    // Removed JS hover effects as CSS handles them now
    // uploadArea.addEventListener('mouseenter', function() {
    //     this.style.transform = 'scale(1.02)';
    // });
    // 
    // uploadArea.addEventListener('mouseleave', function() {
    //     this.style.transform = 'scale(1)';
    // });
});