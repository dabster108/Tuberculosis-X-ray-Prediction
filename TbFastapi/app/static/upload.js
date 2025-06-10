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
    
    let selectedFile = null;
    
    console.log('TB Analyzer JS loaded');
    
    // Event Listeners
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    analyzeBtn.addEventListener('click', analyzeImage);
    resetBtn.addEventListener('click', resetUI);
    
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
    }
    
    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            console.log('File loaded for preview');
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'block';
        };
        
        reader.onerror = function(e) {
            console.error('Error reading file:', e);
            showError('Error reading file. Please try a different image.');
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
                    throw new Error(errorData.detail || 'Network response was not ok');
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
        
        if (data.prediction === 'Normal') {
            resultClass = 'result-normal';
            resultMessage = `<h3 class="${resultClass}">✓ No signs of Tuberculosis detected</h3>
                           <p>The X-ray appears normal. Regular check-ups are still recommended.</p>`;
        } else {
            resultClass = 'result-tb';
            resultMessage = `<h3 class="${resultClass}">⚠ Potential Tuberculosis detected</h3>
                           <p>The analysis indicates possible signs of Tuberculosis. Please consult a medical professional immediately.</p>`;
        }
        
        resultText.innerHTML = resultMessage;
        resultContainer.style.display = 'block';
        
        // Add animation effect
        resultContainer.style.opacity = '0';
        setTimeout(() => {
            resultContainer.style.transition = 'opacity 0.5s ease';
            resultContainer.style.opacity = '1';
        }, 100);
    }
    
    // Reset the UI
    function resetUI() {
        console.log('UI reset');
        selectedFile = null;
        previewImage.src = '';
        uploadArea.style.display = 'block';
        previewContainer.style.display = 'none';
        resultContainer.style.display = 'none';
    }
    
    // Show error message
    function showError(message) {
        console.error('Error:', message);
        alert(message);
    }
    
    // Add visual feedback on hover
    uploadArea.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.02)';
    });
    
    uploadArea.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
    });
});