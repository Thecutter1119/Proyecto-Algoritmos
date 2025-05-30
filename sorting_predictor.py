import numpy as np
from tensorflow.keras import layers, models
from code_vectorizer import CodeVectorizer

class SortingPredictor:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.vectorizer = CodeVectorizer(vector_size=vector_size)
        
        # Tipos de algoritmos de ordenamiento
        self.sorting_algorithms = [
            'bubble_sort',
            'selection_sort', 
            'insertion_sort',
            'merge_sort',
            'quick_sort',
            'heap_sort',
            'counting_sort',
            'radix_sort'
        ]
        
        # Crear el modelo después de definir los algoritmos
        self.model = self._create_model()

    def _create_model(self):
        """Crea el modelo de red neuronal especializado para ordenamiento"""
        model = models.Sequential([
            # Capa de entrada más grande para capturar patrones complejos
            layers.Dense(256, activation='relu', input_shape=(self.vector_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Capas intermedias especializadas
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            
            # Capa de salida para clasificar tipos de ordenamiento
            layers.Dense(len(self.sorting_algorithms), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, code_samples, labels, epochs=20, batch_size=32):
        """Entrena el modelo con códigos de ordenamiento"""
        print("Vectorizando códigos de ordenamiento...")
        vectors = []
        for code in code_samples:
            vector = self.vectorizer.vectorize(code)
            if vector is not None:
                vectors.append(vector)
            else:
                # Vector por defecto si falla la vectorización
                vectors.append(np.zeros(self.vector_size))
        
        vectors = np.array(vectors)
        
        print(f"Entrenando modelo con {len(vectors)} muestras...")
        history = self.model.fit(
            vectors, np.array(labels),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

    def predict(self, code_string):
        """Predice el tipo de algoritmo de ordenamiento"""
        try:
            vector = self.vectorizer.vectorize(code_string)
            if vector is None:
                return None
            
            vector = np.array([vector])
            prediction = self.model.predict(vector, verbose=0)
            
            # Obtener probabilidades para todos los algoritmos
            probabilities = prediction[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            result = {
                'algorithm': self.sorting_algorithms[predicted_idx],
                'confidence': float(confidence),
                'all_probabilities': {
                    alg: float(prob) for alg, prob in 
                    zip(self.sorting_algorithms, probabilities)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            return None

    def evaluate(self, test_code_samples, test_labels):
        """Evalúa el rendimiento del modelo"""
        vectors = []
        for code in test_code_samples:
            vector = self.vectorizer.vectorize(code)
            if vector is not None:
                vectors.append(vector)
            else:
                vectors.append(np.zeros(self.vector_size))
        
        vectors = np.array(vectors)
        evaluation = self.model.evaluate(vectors, np.array(test_labels), verbose=0)
        
        return {
            'loss': evaluation[0],
            'accuracy': evaluation[1]
        }

    def get_algorithm_name(self, index):
        """Convierte índice a nombre del algoritmo"""
        return self.sorting_algorithms[index] if 0 <= index < len(self.sorting_algorithms) else "unknown"