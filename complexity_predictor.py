import numpy as np
from tensorflow.keras import layers, models
from code_vectorizer import CodeVectorizer

class ComplexityPredictor:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.vectorizer = CodeVectorizer(vector_size=vector_size)
        self.o_model = self._create_model()
        self.omega_model = self._create_model()
        self.theta_model = self._create_model()

    def _create_model(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.vector_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(6, activation='softmax')  # 6 clases: O(1), O(log n), O(n), O(n log n), O(n²), O(2^n)
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def code_to_vector(self, code_string):
        """Convierte código fuente a un vector de características usando el vectorizador"""
        return self.vectorizer.vectorize(code_string)

    def train(self, code_samples, o_labels, omega_labels, theta_labels, epochs=10, batch_size=32):
        """Entrena los tres modelos con los datos proporcionados"""
        print("Vectorizando código...")
        vectors = [self.code_to_vector(code) for code in code_samples]
        vectors = np.array(vectors)
        
        print("\nEntrenando modelo para complejidad O...")
        self.o_model.fit(
            vectors, np.array(o_labels),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        print("\nEntrenando modelo para complejidad Ω...")
        self.omega_model.fit(
            vectors, np.array(omega_labels),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        print("\nEntrenando modelo para complejidad Θ...")
        self.theta_model.fit(
            vectors, np.array(theta_labels),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

    def predict(self, code_string):
        """Predice las complejidades O, Ω y Θ para un código dado"""
        vector = self.code_to_vector(code_string)
        if vector is None:
            return None

        vector = np.array([vector])  # Reshape para predicción
        
        o_pred = np.argmax(self.o_model.predict(vector, verbose=0))
        omega_pred = np.argmax(self.omega_model.predict(vector, verbose=0))
        theta_pred = np.argmax(self.theta_model.predict(vector, verbose=0))
        
        return {
            'O': self._complexity_class_to_string(o_pred),
            'Ω': self._complexity_class_to_string(omega_pred),
            'Θ': self._complexity_class_to_string(theta_pred)
        }

    def _complexity_class_to_string(self, class_index):
        """Convierte el índice de clase a su representación en string"""
        classes = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(2^n)']
        return classes[class_index]

    def evaluate(self, test_code_samples, test_o_labels, test_omega_labels, test_theta_labels):
        """Evalúa el rendimiento de los modelos en un conjunto de prueba"""
        vectors = [self.code_to_vector(code) for code in test_code_samples]
        vectors = np.array(vectors)

        o_eval = self.o_model.evaluate(vectors, np.array(test_o_labels), verbose=0)
        omega_eval = self.omega_model.evaluate(vectors, np.array(test_omega_labels), verbose=0)
        theta_eval = self.theta_model.evaluate(vectors, np.array(test_theta_labels), verbose=0)

        return {
            'O': {'loss': o_eval[0], 'accuracy': o_eval[1]},
            'Ω': {'loss': omega_eval[0], 'accuracy': omega_eval[1]},
            'Θ': {'loss': theta_eval[0], 'accuracy': theta_eval[1]}
        }