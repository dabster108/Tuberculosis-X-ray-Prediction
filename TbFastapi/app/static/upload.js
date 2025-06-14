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

    // --- START: Enhancements for "Modern Web" Feel ---
    // Attempt to get common page structure elements (these IDs are hypothetical)
    // Ensure your HTML has elements with these IDs for these features to work.
    const pageTitleElement = document.getElementById('page-title');
    const heroSection = document.getElementById('hero-section');
    const mainContent = document.getElementById('main-content'); // Or a wrapper for the main body
    const footerElement = document.getElementById('footer-element');

    // 1. Welcome Message & Animation
    if (pageTitleElement) {
        pageTitleElement.textContent = 'Welcome to Tuberculosis X-Ray Analysis';
        pageTitleElement.style.opacity = '0';
        pageTitleElement.style.transition = 'opacity 0.8s ease-in-out 0.2s';
        setTimeout(() => pageTitleElement.style.opacity = '1', 50);
    } else {
        console.info('Modern UI Tip: Add an element with id="page-title" in your HTML to display a welcome message.');
    }

    // 2. Initial Animations for key page elements
    // For more complex animations, prefer using CSS classes and transitions/animations.
    if (uploadArea) {
        uploadArea.style.opacity = '0';
        uploadArea.style.transform = 'translateY(20px)';
        uploadArea.style.transition = 'opacity 0.6s ease-out 0.4s, transform 0.6s ease-out 0.4s';
        setTimeout(() => {
            uploadArea.style.opacity = '1';
            uploadArea.style.transform = 'translateY(0)';
        }, 100);
    }

    if (heroSection) {
        heroSection.style.opacity = '0';
        heroSection.style.transition = 'opacity 1s ease-in-out 0.1s';
        setTimeout(() => heroSection.style.opacity = '1', 50);
    }

    if (mainContent) {
        mainContent.style.opacity = '0';
        mainContent.style.transition = 'opacity 1s ease-in-out 0.6s';
        setTimeout(() => mainContent.style.opacity = '1', 50);
    }

    if (footerElement) {
        footerElement.style.opacity = '0';
        footerElement.style.transition = 'opacity 1s ease-in-out 0.9s';
        setTimeout(() => footerElement.style.opacity = '1', 50);
    }
    // --- END: Enhancements for "Modern Web" Feel ---

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
        // analyzeBtn.disabled = false; // Keep this commented or remove, display style is used instead
    }

    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            console.log('File loaded for preview');
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'block'; 
            
            // Show Analyze and Reset buttons
            if(analyzeBtn) analyzeBtn.style.display = 'inline-block';
            if(resetBtn) resetBtn.style.display = 'inline-block';

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
        previewContainer.classList.remove('visible'); 
        resultContainer.style.display = 'none';
        resultContainer.classList.remove('show');
        fileInput.value = '';

        // Hide Analyze and Reset buttons
        if(analyzeBtn) analyzeBtn.style.display = 'none';
        if(resetBtn) resetBtn.style.display = 'none';
        if(loadingContainer) loadingContainer.style.display = 'none'; // Ensure loading is also hidden on reset
    }

    // Show error message
    function showError(message) {
        console.error('UI Error Display:', message);

        if (loadingContainer) {
            loadingContainer.style.display = 'none'; // Ensure loading indicator is hidden
        }

        if (resultText && resultContainer) {
            resultText.innerHTML = ''; // Clear previous messages
            resultText.className = 'result-text'; // Reset to base class

            const errorMessageElement = document.createElement('p');
            errorMessageElement.className = 'error-message'; // For specific error styling via CSS
            errorMessageElement.textContent = message;
            // Basic inline styling for immediate visibility, ideally use CSS class:
            errorMessageElement.style.color = '#D8000C'; // Error red
            errorMessageElement.style.fontWeight = 'bold';
            resultText.appendChild(errorMessageElement);

            resultContainer.style.display = 'block';
            resultContainer.classList.remove('show');
            // Force reflow for transition re-trigger
            void resultContainer.offsetWidth; 
            setTimeout(() => {
                resultContainer.classList.add('show');
            }, 10);
        } else {
            console.warn('Result container or text element not found for displaying error. Using alert.');
            alert(message); // Fallback
        }
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