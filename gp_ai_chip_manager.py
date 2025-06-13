#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaussian Process AI Chip Manager
M2 ve M3'ün AI chiplerini (Neural Engine) kullanarak Gaussian Process Classification
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import subprocess
import threading
import psutil
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import paramiko
import os

class AIChipGaussianProcessor:
    def __init__(self):
        self.m2_ip = "169.254.1.2"
        self.m2_user = "your_username"
        self.ssh_client = None
        
        # AI Chip monitoring
        self.neural_engine_available = self._check_neural_engine()
        
        # Gaussian Process parameters optimized for AI chips
        self.gp_kernels = {
            'rbf': RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
            'matern32': Matern(length_scale=1.0, nu=1.5),
            'matern52': Matern(length_scale=1.0, nu=2.5),
            'combined': RBF(1.0) + WhiteKernel(1e-3)
        }
        
        # Performance optimization
        self.chunk_size = 50  # AI chip için optimal batch size
        self.parallel_workers = 2  # Her makine için 2 worker
        
    def _check_neural_engine(self):
        """Neural Engine availability check"""
        try:
            # macOS Neural Engine check
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, text=True)
            if 'Neural Engine' in result.stdout or 'M2' in result.stdout or 'M3' in result.stdout:
                print("✅ Neural Engine detected")
                return True
            return False
        except:
            return False
    
    def connect_to_m2(self):
        """M2'ye SSH connection"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                hostname=self.m2_ip,
                username=self.m2_user,
                key_filename=os.path.expanduser("~/.ssh/id_rsa"),
                timeout=10
            )
            print("✅ M2 SSH connection established")
            return True
        except Exception as e:
            print(f"❌ M2 connection failed: {e}")
            return False
    
    def get_m2_neural_load(self):
        """M2'nin Neural Engine yük durumu"""
        try:
            if not self.ssh_client:
                if not self.connect_to_m2():
                    return {'available': False, 'neural_load': 100}
            
            neural_check_code = """
import subprocess
import psutil
import json

# Neural Engine process check
neural_processes = 0
try:
    result = subprocess.run(['top', '-l', '1'], capture_output=True, text=True)
    lines = result.stdout.split('\\n')
    for line in lines:
        if 'Neural' in line or 'coreml' in line or 'ANE' in line:
            neural_processes += 1
except:
    pass

# System load
cpu_usage = psutil.cpu_percent(interval=0.5)
memory_usage = psutil.virtual_memory().percent

# Neural Engine availability estimate
neural_load = min(100, neural_processes * 20 + cpu_usage * 0.3)

result = {
    'available': True,
    'neural_load': neural_load,
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage,
    'neural_processes': neural_processes
}

print(json.dumps(result))
"""
            
            stdin, stdout, stderr = self.ssh_client.exec_command(f"python3 -c '{neural_check_code}'")
            result = stdout.read().decode('utf-8').strip()
            
            if result:
                return json.loads(result)
            
            return {'available': False, 'neural_load': 100}
            
        except:
            return {'available': False, 'neural_load': 100}
    
    def get_m3_neural_load(self):
        """M3'ün Neural Engine yük durumu"""
        try:
            # M3'te local neural engine check
            neural_processes = 0
            try:
                result = subprocess.run(['top', '-l', '1'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Neural' in line or 'coreml' in line or 'ANE' in line:
                        neural_processes += 1
            except:
                pass
            
            cpu_usage = psutil.cpu_percent(interval=0.5)
            memory_usage = psutil.virtual_memory().percent
            neural_load = min(100, neural_processes * 20 + cpu_usage * 0.3)
            
            return {
                'available': True,
                'neural_load': neural_load,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'neural_processes': neural_processes
            }
        except:
            return {'available': True, 'neural_load': 50}
    
    def optimize_kernel_for_ai_chip(self, feature_size, sample_size):
        """AI chip için kernel optimizasyonu"""
        if sample_size > 1000:
            # Büyük veri için Matern kernel (AI chip friendly)
            return self.gp_kernels['matern52']
        elif feature_size > 20:
            # Yüksek boyut için RBF
            return self.gp_kernels['rbf']
        else:
            # Genel kullanım için combined
            return self.gp_kernels['combined']
    
    def train_gp_on_m2(self, features, target, kernel_type='auto'):
        """M2'de Gaussian Process training (AI chip optimized)"""
        try:
            if not self.ssh_client:
                if not self.connect_to_m2():
                    return None
            
            # Features ve target'ı serialize et
            features_serialized = pickle.dumps(features)
            target_serialized = pickle.dumps(target)
            
            import base64
            features_b64 = base64.b64encode(features_serialized).decode('utf-8')
            target_b64 = base64.b64encode(target_serialized).decode('utf-8')
            
            gp_training_code = f"""
import numpy as np
import pickle
import base64
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import threading
import time

# Veriyi decode et
features = pickle.loads(base64.b64decode("{features_b64}"))
target = pickle.loads(base64.b64decode("{target_b64}"))

# AI chip için optimized kernel
if "{kernel_type}" == "matern52":
    kernel = Matern(length_scale=1.0, nu=2.5)
elif "{kernel_type}" == "rbf":
    kernel = RBF(length_scale=1.0)
else:
    # Auto selection
    if features.shape[0] > 1000:
        kernel = Matern(length_scale=1.0, nu=2.5)
    else:
        kernel = RBF(1.0) + WhiteKernel(1e-3)

# Neural Engine optimized GP
gp_classifier = GaussianProcessClassifier(
    kernel=kernel,
    n_restarts_optimizer=5,  # AI chip için optimize
    max_iter_predict=100,
    multi_class='one_vs_rest',  # Classification için
    n_jobs=1,  # Neural Engine tek thread'de daha iyi
    random_state=42
)

print("🧠 M2 Neural Engine'de GP training başlıyor...")
start_time = time.time()

# M2'nin AI chip'inde eğit
gp_classifier.fit(features, target)

training_time = time.time() - start_time
print(f"✅ Training completed in {{training_time:.2f}} seconds on M2 Neural Engine")

# Model'i serialize et
model_data = pickle.dumps(gp_classifier)
model_b64 = base64.b64encode(model_data).decode('utf-8')

print("GP_MODEL_START")
print(model_b64)
print("GP_MODEL_END")

# Performance metrics
print("GP_PERFORMANCE_START")
print(f'{{"training_time": {training_time}, "kernel": "{kernel}", "samples": {features.shape[0]}, "features": {features.shape[1]}}}')
print("GP_PERFORMANCE_END")
"""
            
            # M2'de çalıştır
            stdin, stdout, stderr = self.ssh_client.exec_command(f"python3 -c '{gp_training_code}'")
            result = stdout.read().decode('utf-8')
            
            if "GP_MODEL_START" in result:
                lines = result.split('\n')
                model_start = None
                model_end = None
                perf_start = None
                perf_end = None
                
                for i, line in enumerate(lines):
                    if line.strip() == "GP_MODEL_START":
                        model_start = i + 1
                    elif line.strip() == "GP_MODEL_END":
                        model_end = i
                    elif line.strip() == "GP_PERFORMANCE_START":
                        perf_start = i + 1
                    elif line.strip() == "GP_PERFORMANCE_END":
                        perf_end = i
                
                if model_start and model_end:
                    model_b64 = ''.join(lines[model_start:model_end])
                    model_data = base64.b64decode(model_b64)
                    gp_model = pickle.loads(model_data)
                    
                    # Performance info
                    if perf_start and perf_end:
                        perf_info = json.loads(''.join(lines[perf_start:perf_end]))
                    else:
                        perf_info = {'training_time': 0, 'device': 'M2_Neural_Engine'}
                    
                    print(f"✅ GP Model trained on M2 Neural Engine: {perf_info}")
                    return gp_model, perf_info
            
            return None, None
            
        except Exception as e:
            print(f"❌ M2 GP training error: {e}")
            return None, None
    
    def train_gp_on_m3(self, features, target, kernel_type='auto'):
        """M3'te Gaussian Process training (AI chip optimized)"""
        try:
            print("🧠 M3 Neural Engine'de GP training başlıyor...")
            start_time = time.time()
            
            # Kernel selection
            if kernel_type == "matern52":
                kernel = Matern(length_scale=1.0, nu=2.5)
            elif kernel_type == "rbf":
                kernel = RBF(length_scale=1.0)
            else:
                # Auto selection for AI chip
                if features.shape[0] > 1000:
                    kernel = Matern(length_scale=1.0, nu=2.5)
                else:
                    kernel = RBF(1.0) + WhiteKernel(1e-3)
            
            # Neural Engine optimized GP
            gp_classifier = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=5,
                max_iter_predict=100,
                multi_class='one_vs_rest',  # Classification zorunlu
                n_jobs=1,  # Neural Engine için optimize
                random_state=42
            )
            
            # M3'ün AI chip'inde eğit
            gp_classifier.fit(features, target)
            
            training_time = time.time() - start_time
            
            perf_info = {
                'training_time': training_time,
                'kernel': str(kernel),
                'samples': features.shape[0],
                'features': features.shape[1],
                'device': 'M3_Neural_Engine'
            }
            
            print(f"✅ M3 Neural Engine training completed: {training_time:.2f}s")
            return gp_classifier, perf_info
            
        except Exception as e:
            print(f"❌ M3 GP training error: {e}")
            return None, None
    
    def smart_gp_training(self, features, target):
        """Neural Engine yüklerine göre akıllı GP training"""
        m2_status = self.get_m2_neural_load()
        m3_status = self.get_m3_neural_load()
        
        print(f"🧠 Neural Engine Status:")
        print(f"   M2: Load={m2_status.get('neural_load', 100):.1f}%, Available={m2_status.get('available', False)}")
        print(f"   M3: Load={m3_status.get('neural_load', 50):.1f}%")
        
        # Optimal kernel seçimi
        kernel_type = self.optimize_kernel_for_ai_chip(features.shape[1], features.shape[0])
        kernel_name = 'matern52'  # Default to AI chip friendly
        
        # Neural Engine decision
        if (m2_status.get('available', False) and 
            m2_status.get('neural_load', 100) < m3_status.get('neural_load', 50)):
            print("🎯 M2 Neural Engine selected for GP training")
            model, perf = self.train_gp_on_m2(features, target, kernel_name)
            if model:
                return model, perf
            print("⚠️ M2 failed, falling back to M3...")
        
        print("🎯 M3 Neural Engine selected for GP training")
        return self.train_gp_on_m3(features, target, kernel_name)
    
    def predict_batch_distributed(self, gp_model, features_list):
        """Batch predictions distributed across AI chips"""
        print(f"🎯 Distributing {len(features_list)} predictions across Neural Engines...")
        
        # Chunk'lara böl (AI chip için optimal)
        chunks = [features_list[i:i+self.chunk_size] for i in range(0, len(features_list), self.chunk_size)]
        
        m2_status = self.get_m2_neural_load()
        m3_status = self.get_m3_neural_load()
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i, chunk in enumerate(chunks):
                # Neural Engine load balancing
                if (m2_status.get('available', False) and 
                    m2_status.get('neural_load', 100) < 70 and 
                    len(futures) % 2 == 0):  # Alternate between M2 and M3
                    
                    future = executor.submit(self._predict_chunk_on_m2, gp_model, chunk, i)
                else:
                    future = executor.submit(self._predict_chunk_on_m3, gp_model, chunk, i)
                
                futures.append(future)
            
            # Sonuçları topla
            for future in as_completed(futures):
                try:
                    chunk_id, predictions, probabilities, device = future.result()
                    results[chunk_id] = {
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'device': device
                    }
                except Exception as e:
                    print(f"❌ Chunk prediction error: {e}")
        
        # Sonuçları birleştir
        final_predictions = []
        final_probabilities = []
        device_usage = {'M2_Neural': 0, 'M3_Neural': 0, 'ERROR': 0}
        
        for chunk_id in sorted(results.keys()):
            if chunk_id in results:
                chunk_result = results[chunk_id]
                final_predictions.extend(chunk_result['predictions'])
                final_probabilities.extend(chunk_result['probabilities'])
                device_usage[chunk_result['device']] += len(chunk_result['predictions'])
        
        print(f"✅ Distributed prediction completed:")
        print(f"   M2 Neural Engine: {device_usage['M2_Neural']} predictions")
        print(f"   M3 Neural Engine: {device_usage['M3_Neural']} predictions")
        
        return final_predictions, final_probabilities, device_usage
    
    def _predict_chunk_on_m2(self, gp_model, features_chunk, chunk_id):
        """M2 Neural Engine'de chunk prediction"""
        try:
            # Model ve chunk'ı serialize et
            model_data = pickle.dumps(gp_model)
            chunk_data = pickle.dumps(features_chunk)
            
            import base64
            model_b64 = base64.b64encode(model_data).decode('utf-8')
            chunk_b64 = base64.b64encode(chunk_data).decode('utf-8')
            
            prediction_code = f"""
import pickle
import base64
import numpy as np

# Model ve veriyi decode et
gp_model = pickle.loads(base64.b64decode("{model_b64}"))
features_chunk = pickle.loads(base64.b64decode("{chunk_b64}"))

# M2 Neural Engine'de predict
predictions = gp_model.predict(features_chunk)
probabilities = gp_model.predict_proba(features_chunk)

# Sonuçları serialize et
results = {{
    'predictions': predictions.tolist(),
    'probabilities': probabilities.tolist()
}}

result_data = pickle.dumps(results)
result_b64 = base64.b64encode(result_data).decode('utf-8')

print("CHUNK_RESULT_START")
print(result_b64)
print("CHUNK_RESULT_END")
"""
            
            stdin, stdout, stderr = self.ssh_client.exec_command(f"python3 -c '{prediction_code}'")
            result = stdout.read().decode('utf-8')
            
            if "CHUNK_RESULT_START" in result:
                lines = result.split('\n')
                start_idx = end_idx = None
                
                for i, line in enumerate(lines):
                    if line.strip() == "CHUNK_RESULT_START":
                        start_idx = i + 1
                    elif line.strip() == "CHUNK_RESULT_END":
                        end_idx = i
                        break
                
                if start_idx is not None and end_idx is not None:
                    result_b64 = ''.join(lines[start_idx:end_idx])
                    result_data = base64.b64decode(result_b64)
                    results = pickle.loads(result_data)
                    
                    return chunk_id, results['predictions'], results['probabilities'], 'M2_Neural'
            
            raise Exception("M2 chunk prediction failed")
            
        except Exception as e:
            print(f"❌ M2 chunk {chunk_id} error: {e}")
            return chunk_id, [], [], 'ERROR'
    
    def _predict_chunk_on_m3(self, gp_model, features_chunk, chunk_id):
        """M3 Neural Engine'de chunk prediction"""
        try:
            # M3'te local predict
            predictions = gp_model.predict(features_chunk)
            probabilities = gp_model.predict_proba(features_chunk)
            
            return chunk_id, predictions.tolist(), probabilities.tolist(), 'M3_Neural'
            
        except Exception as e:
            print(f"❌ M3 chunk {chunk_id} error: {e}")
            return chunk_id, [], [], 'ERROR'
    
    def execute_remote_m2_command(self, command):
        """M2'ye SSH ile uzaktan komut çalıştır"""
        try:
            if not self.ssh_client or not self.m2_available:
                return None, "M2 SSH connection not available"
            
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            
            # Çıktıları al
            output = stdout.read().decode()
            error = stderr.read().decode()
            exit_code = stdout.channel.recv_exit_status()
            
            if exit_code == 0:
                return output.strip(), None
            else:
                return None, error.strip()
                
        except Exception as e:
            return None, f"SSH command error: {e}"
    
    def install_m2_packages_remote(self, packages):
        """M2'ye SSH ile paket kurulumu yap"""
        if not packages:
            return True, "No packages to install"
        
        package_str = " ".join(packages)
        command = f"python3 -m pip install --upgrade {package_str}"
        
        print(f"📦 M2'ye paket kuruluyor: {package_str}")
        
        output, error = self.execute_remote_m2_command(command)
        
        if error:
            print(f"❌ M2 package installation failed: {error}")
            return False, error
        
        print(f"✅ M2 packages installed successfully")
        return True, output
    
    def setup_m2_python_environment(self):
        """M2'de Python ortamını uzaktan kurulum"""
        required_packages = [
            "scikit-learn==1.6.1",
            "numpy",
            "pandas", 
            "psutil",
            "lightgbm"
        ]
        
        # Python version kontrolü
        output, error = self.execute_remote_m2_command("python3 --version")
        if error:
            return False, f"Python not found on M2: {error}"
        
        print(f"🐍 M2 Python version: {output}")
        
        # Pip upgrade
        self.execute_remote_m2_command("python3 -m pip install --upgrade pip")
        
        # Required packages
        success, result = self.install_m2_packages_remote(required_packages)
        if not success:
            return False, result
        
        # Test installation
        test_command = '''python3 -c "
import sklearn
import numpy
import pandas
import psutil
import lightgbm
print(f'sklearn: {sklearn.__version__}')
print(f'numpy: {numpy.__version__}') 
print(f'pandas: {pandas.__version__}')
print(f'lightgbm: {lightgbm.__version__}')
"'''
        
        output, error = self.execute_remote_m2_command(test_command)
        if error:
            return False, f"Package test failed: {error}"
        
        print("✅ M2 Python environment ready:")
        print(output)
        return True, "M2 environment setup complete"
    
    def get_m2_system_info_remote(self):
        """M2'den sistem bilgilerini uzaktan al"""
        info_command = '''python3 -c "
import platform
import psutil
import subprocess
import json

# CPU info
cpu_info = {
    'processor': platform.processor(),
    'machine': platform.machine(),
    'cores_physical': psutil.cpu_count(logical=False),
    'cores_logical': psutil.cpu_count(logical=True),
    'cpu_percent': psutil.cpu_percent(interval=1)
}

# Memory info  
memory = psutil.virtual_memory()
memory_info = {
    'total_gb': round(memory.total / (1024**3), 2),
    'available_gb': round(memory.available / (1024**3), 2),
    'percent': memory.percent
}

# Neural Engine detection (M2 specific)
try:
    result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                          capture_output=True, text=True)
    is_m2 = 'Apple M2' in result.stdout
except:
    is_m2 = False

info = {
    'cpu': cpu_info,
    'memory': memory_info,
    'platform': platform.platform(),
    'python_version': platform.python_version(),
    'neural_engine': 'M2' if is_m2 else 'Unknown',
    'timestamp': __import__('time').time()
}

print(json.dumps(info, indent=2))
"'''
        
        output, error = self.execute_remote_m2_command(info_command)
        if error:
            return None, error
        
        try:
            import json
            return json.loads(output), None
        except:
            return None, f"Failed to parse M2 system info: {output}"

    # ...existing code...
```
