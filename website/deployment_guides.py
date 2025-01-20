class DeploymentGuideGenerator:
    def __init__(self, model_info, pipeline_type):
        self.model_info = model_info
        self.pipeline_type = pipeline_type

    def generate_local_guide(self):
        """Generate instructions for local deployment"""
        return f"""## Local Deployment Guide

### Prerequisites
- Docker installed on your machine
- Python 3.9 or later
- At least {self.estimate_disk_space()} of free disk space

### Steps

1. **Extract the Module Files**
   ```bash
   unzip lilypad-module.zip
   cd lilypad-module
   ```

2. **Build the Docker Image**
   ```bash
   docker build -t {self.model_info.id.split('/')[-1]} .
   ```

3. **Prepare Your Input**
   {self._get_input_preparation_guide()}

4. **Run the Module**
   ```bash
   docker run -v $(pwd)/input:/workspace/input \\
             -v $(pwd)/output:/outputs \\
             -e INPUT_PATH=/workspace/input/{self._get_input_filename()} \\
             {self.model_info.id.split('/')[-1]}
   ```

5. **Check Results**
   The results will be available in the `output/result.json` file.
"""

    def generate_lilypad_guide(self):
        """Generate instructions for Lilypad deployment"""
        return f"""## Lilypad Deployment Guide

### Prerequisites
- Lilypad CLI installed
- Access to Lilypad network
- Module files downloaded and extracted

### Steps

1. **Login to Lilypad**
   ```bash
   lilypad login
   ```

2. **Deploy the Module**
   ```bash
   cd lilypad-module
   lilypad module deploy .
   ```

3. **Run on Lilypad**
   ```bash
   lilypad run {self.model_info.id.split('/')[-1]} \\
       --input "INPUT_PATH=/path/to/input/{self._get_input_filename()}"
   ```

4. **Monitor Progress**
   ```bash
   lilypad status <job-id>
   ```

5. **Get Results**
   ```bash
   lilypad get-results <job-id>
   ```
"""

    def generate_kubernetes_guide(self):
        """Generate instructions for Kubernetes deployment"""
        return f"""## Kubernetes Deployment Guide

### Prerequisites
- Kubernetes cluster
- kubectl configured
- Docker registry access

### Steps

1. **Build and Push the Image**
   ```bash
   docker build -t your-registry/{self.model_info.id.split('/')[-1]} .
   docker push your-registry/{self.model_info.id.split('/')[-1]}
   ```

2. **Create Kubernetes Manifests**
   Create a file named `deployment.yaml`:

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: {self.model_info.id.split('/')[-1]}
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: {self.model_info.id.split('/')[-1]}
     template:
       metadata:
         labels:
           app: {self.model_info.id.split('/')[-1]}
       spec:
         containers:
         - name: model
           image: your-registry/{self.model_info.id.split('/')[-1]}
           resources:
             requests:
               memory: "4Gi"
               cpu: "1"
             limits:
               memory: "8Gi"
               cpu: "2"
           volumeMounts:
           - name: input-volume
             mountPath: /workspace/input
           - name: output-volume
             mountPath: /outputs
         volumes:
         - name: input-volume
           persistentVolumeClaim:
             claimName: input-pvc
         - name: output-volume
           persistentVolumeClaim:
             claimName: output-pvc
   ```

3. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f deployment.yaml
   ```

4. **Monitor the Deployment**
   ```bash
   kubectl get pods
   kubectl logs -f deployment/{self.model_info.id.split('/')[-1]}
   ```
"""

    def _get_input_preparation_guide(self):
        """Get input preparation instructions based on pipeline type"""
        guides = {
            'text-classification': """Create a text file with your input:
   ```bash
   echo "Your text to classify" > input/input.txt
   ```""",
            'image-classification': """Place your image file in the input directory:
   ```bash
   cp your-image.jpg input/input.jpg
   ```""",
            'object-detection': """Place your image file in the input directory:
   ```bash
   cp your-image.jpg input/input.jpg
   ```""",
            'question-answering': """Create a JSON file with your question and context:
   ```bash
   echo '{"question": "Your question", "context": "Your context"}' > input/input.json
   ```"""
        }
        return guides.get(self.pipeline_type, """Prepare your input file according to the model requirements:
   ```bash
   cp your-input-file input/input.txt
   ```""")

    def _get_input_filename(self):
        """Get default input filename based on pipeline type"""
        extensions = {
            'text-classification': 'txt',
            'image-classification': 'jpg',
            'object-detection': 'jpg',
            'question-answering': 'json'
        }
        return f"input.{extensions.get(self.pipeline_type, 'txt')}"

    def estimate_disk_space(self):
        """Estimate required disk space based on model size"""
        model_size_mb = self.model_info.size_in_bytes / (1024 * 1024)
        # Add space for Docker image and dependencies
        total_size_gb = (model_size_mb + 500) / 1024  # Add 500MB for Docker image and deps
        return f"{total_size_gb:.1f}GB"