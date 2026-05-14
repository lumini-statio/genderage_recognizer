import numpy as np
import cv2
from insightface.app import FaceAnalysis
from gender_guesser.detector import Detector
import pickle
import os
import matplotlib.pyplot as plt

class GenderAnalysis:
    def __init__(self, app: FaceAnalysis, gamma: float = 1.0):
        self.app = app
        self.gamma = gamma
        # Pre-calculamos la tabla LUT para la corrección gamma
        self.lut = self._build_lut(gamma)
        self.gender_guesser = Detector()

    def _build_lut(self, gamma: float):
        """Crea una tabla de búsqueda para aplicar gamma rápidamente."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        return table

    def apply_gamma(self, img: np.ndarray) -> np.ndarray:
        """Aplica la corrección gamma a la imagen usando la LUT."""
        if self.gamma == 1.0:
            return img
        return cv2.LUT(img, self.lut)

    def get_embedding(self, img_path: str) -> np.ndarray | None:
        """Detecta embeddings solo si hay EXACTAMENTE una cara en la imagen.
        Descarta imágenes con 0 o múltiples caras.
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f'{img_path}: no se pudo leer la imagen')
            return None
        
        # Aplicamos corrección gamma antes de la detección
        img_processed = self.apply_gamma(img)
        
        faces = self.app.get(img_processed)
        if faces and len(faces) == 1:
            face = faces[0]
            gender = 'hombre' if face.gender == 1 else 'mujer'
            return face.normed_embedding, gender, face.age
        
        if faces and len(faces) > 1:
            print(f'{img_path}: se detectaron múltiples caras ({len(faces)}), descartada')
        else:
            print(f'{img_path}: no se reconoció ninguna cara en la imagen')
        return None

    def process_and_save_errors(self, dir_path: str, output_dir: str = 'out'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            img = cv2.imread(file_path)
            if img is None: continue

            # Procesamos la imagen para la inferencia
            img_processed = self.apply_gamma(img)
            faces = self.app.get(img_processed)
            
            if not faces: continue

            fn_lower = filename.lower()
            real_gender = 'hombre' if 'hombre' in fn_lower else 'mujer' if 'mujer' in fn_lower else None
            if real_gender is None: continue

            prediction_error = False
            for face in faces:
                bbox = face.bbox.astype(int)
                pred_gender = 'hombre' if face.gender == 1 else 'mujer'
                
                # Dibujamos sobre la imagen ORIGINAL (para que el output se vea natural)
                # o sobre la procesada si prefieres ver el efecto del gamma.
                color = (0, 0, 255)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                label = f"{pred_gender}"
                cv2.putText(img, label, (bbox[0], bbox[3] + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if pred_gender != real_gender:
                    prediction_error = True

            if prediction_error:
                out_path = os.path.join(output_dir, f'err_{filename}')
                cv2.imwrite(out_path, img)


    def embed_all(self, dir_path: str, pickle_filename: str):
        # ... (el resto del código se mantiene igual ya que llama a get_embedding)
        embeddings, genders, ages, files = [], [], [], []
        
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ['.jpg', '.png', '.jpeg']:
                    continue
                
                file_path = os.path.join(root, filename)
                res = self.get_embedding(file_path)
                
                if res:
                    emb, gen, age = res
                    embeddings.append(emb)
                    genders.append(gen)
                    ages.append(age)
                    files.append(file_path)

        data = {'embeddings': embeddings, 'files': files, 'genders': genders, 'ages': ages}
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)
        
    
    def get_all_embeddings(self, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        data['records'] = [
            {
                'embedding': embedding,
                'filename': filename,
                'age': age,
                'gender': gender
            }
            for embedding, filename, age, gender in zip(data['embeddings'], data['files'], data['ages'], data['genders'])
        ]

        return data

    
    def evaluate_gender_accuracy(self):
        exitos = []
        erroneos = []
        data = self.get_all_embeddings('people.pkl')
        
        for record in data['records']:
            # Extraer género real del nombre del archivo
            filename = record['filename'].lower()
            if 'hombre' in filename:
                real_gender = 'hombre'
            elif 'mujer' in filename:
                real_gender = 'mujer'
            else:
                print('no hay ni HOMBRE ni MUJER en el nombre del archivo')
                continue
            
            if record['gender'] == real_gender:
                exitos.append(record)
            else:
                erroneos.append(record)
        
        total = len(exitos) + len(erroneos)
        pct_exitos = (len(exitos) / total) * 100 if total > 0 else 0
        pct_erroneos = (len(erroneos) / total) * 100 if total > 0 else 0
        
        return (len(exitos), len(erroneos), pct_exitos, pct_erroneos)
    

    def evaluate_lfw_gender_accuracy(self):
        import requests, time
        exitos = []
        erroneos = []
        
        data = self.get_all_embeddings('lfw_genres_insightface.pkl')
        
        for record in data['records']:
            try:
                filename: str = record['filename'].split('/')[-1]
                person_name: str = filename.split('_')[0]
                guessed_gender = self.gender_guesser.get_gender(person_name)                
                # Convertir a español para comparar
                if guessed_gender == 'male':
                    expected_gender = 'hombre'
                elif guessed_gender == 'female':
                    expected_gender = 'mujer'
                    
                if record['gender'] == expected_gender: 
                    exitos.append(record)
                else:
                    erroneos.append(record)
            except Exception as e:
                print(f"Error consultando {person_name}: {e}")

        total = len(exitos) + len(erroneos)
        pct_exitos = (len(exitos) / total) * 100 if total > 0 else 0
        pct_erroneos = (len(erroneos) / total) * 100 if total > 0 else 0

        # Crear carpeta de salida
        if not os.path.exists('out'):
            os.makedirs('out')

        for record in erroneos[:10]:
            try:
                img = cv2.imread(record['filename'])
                if img is None:
                    print(f"No se pudo leer: {record['filename']}")
                    continue
                
                faces = self.app.get(img)
                if not faces:
                    print(f"No se detectaron caras en: {record['filename']}")
                    continue

                bbox = faces[0].bbox.astype(int)
                color = (0, 0, 255)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f"{record['gender']}"
                cv2.putText(img, label, (bbox[0], bbox[3] + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                out_filename = record['filename'].split('/')[-1]
                out_path = os.path.join('out', f'err_{out_filename}')
                cv2.imwrite(out_path, img)
                print(f"Guardada: {out_path}")
            except Exception as e:
                print(f"Error procesando {record['filename']}: {e}")

        return (len(exitos), len(erroneos), pct_exitos, pct_erroneos)


    def plot_gender_accuracy(self, results):
        exitos_len, erroneos_len, pct_exitos, pct_erroneos = results
        
        labels = ['Éxitos', 'Errores']
        values = [pct_exitos, pct_erroneos]
        colors = ['blue', 'red']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, values, color=colors)
        
        # Configuración de ejes y etiquetas
        plt.ylabel('Porcentaje (%)')
        plt.title('Porcentaje Detección de Género InsightFace')
        plt.ylim(0, 100)  # Asegura la escala de 0 a 100
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()