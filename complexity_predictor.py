import numpy as np
import re
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from code_vectorizer import CodeVectorizer

class ComplexityPredictor:
    def __init__(self, vector_size=100, num_classes=5):
        self.vector_size = vector_size
        self.num_classes = num_classes
        self.vectorizer = CodeVectorizer(vector_size=vector_size)
        
        # Modelos especializados
        self.o_model = None
        self.omega_model = None
        self.theta_model = None
        
        # Mapeo de complejidades corregido
        self.complexity_map = {
            0: 'O(1)',
            1: 'O(log n)', 
            2: 'O(n)',
            3: 'O(n log n)',
            4: 'O(n²)'
        }
        
        # Patrones de código para análisis mejorado
        self.code_patterns = {
            'binary_search': [
                r'left.*right.*mid',
                r'binary.*search',
                r'while.*left.*<=.*right',
                r'mid.*=.*(left.*\+.*right).*//.*2'
            ],
            'linear_search': [
                r'for.*in.*range.*len',
                r'linear.*search',
                r'if.*arr\[i\].*==.*target'
            ],
            'bubble_sort': [
                r'bubble.*sort',
                r'for.*range.*n.*for.*range.*n-i',
                r'arr\[j\].*>.*arr\[j\+1\]'
            ],
            'merge_sort': [
                r'merge.*sort',
                r'merge\(.*left.*right\)',
                r'arr\[:mid\].*arr\[mid:\]'
            ],
            'quick_sort': [
                r'quick.*sort',
                r'partition',
                r'pivot'
            ]
        }

    def _detect_algorithm_type(self, code_string):
        """Detecta el tipo de algoritmo basado en patrones"""
        code_lower = code_string.lower()
        
        for algo_type, patterns in self.code_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_lower):
                    return algo_type
        return 'unknown'

    def _get_omega_features(self, code_string, base_vector):
        """Características específicas para Ω (mejor caso)"""
        additional_features = []
        algo_type = self._detect_algorithm_type(code_string)
        
        # Vector one-hot para tipo de algoritmo (mejor caso)
        if algo_type == 'binary_search':
            additional_features.extend([1, 0, 0, 0, 0])  # O(1) - encontrar inmediatamente
        elif algo_type == 'linear_search':
            additional_features.extend([1, 0, 0, 0, 0])  # O(1) - primer elemento
        elif algo_type == 'bubble_sort':
            # Verificar si está optimizado
            if 'swapped' in code_string or 'break' in code_string:
                additional_features.extend([0, 0, 1, 0, 0])  # O(n) optimizado
            else:
                additional_features.extend([0, 0, 0, 0, 1])  # O(n²) sin optimizar
        elif algo_type == 'merge_sort':
            additional_features.extend([0, 0, 0, 1, 0])  # O(n log n) - no cambia
        elif algo_type == 'quick_sort':
            additional_features.extend([0, 1, 0, 0, 0])  # O(log n) - mejor partición
        else:
            # Análisis general para mejor caso
            if self._has_early_return(code_string):
                additional_features.extend([1, 0, 0, 0, 0])  # O(1)
            else:
                additional_features.extend([0, 0, 0, 0, 0])  # Sin información
        
        # Características adicionales para Omega
        additional_features.extend([
            1 if self._has_early_return(code_string) else 0,
            1 if 'break' in code_string else 0,
            1 if self._is_sorted_check(code_string) else 0
        ])
        
        return np.concatenate([base_vector, additional_features])

    def _get_theta_features(self, code_string, base_vector):
        """Características específicas para Θ (caso promedio)"""
        additional_features = []
        algo_type = self._detect_algorithm_type(code_string)
        
        # Vector one-hot para tipo de algoritmo (caso promedio)
        if algo_type == 'binary_search':
            additional_features.extend([0, 1, 0, 0, 0])  # O(log n)
        elif algo_type == 'linear_search':
            additional_features.extend([0, 0, 1, 0, 0])  # O(n)
        elif algo_type == 'bubble_sort':
            additional_features.extend([0, 0, 0, 0, 1])  # O(n²)
        elif algo_type == 'merge_sort':
            additional_features.extend([0, 0, 0, 1, 0])  # O(n log n)
        elif algo_type == 'quick_sort':
            additional_features.extend([0, 0, 0, 1, 0])  # O(n log n) promedio
        else:
            # Análisis general basado en estructura
            loop_depth = self._count_nested_loops(code_string)
            if loop_depth == 0:
                additional_features.extend([1, 0, 0, 0, 0])  # O(1)
            elif loop_depth == 1:
                additional_features.extend([0, 0, 1, 0, 0])  # O(n)
            elif loop_depth >= 2:
                additional_features.extend([0, 0, 0, 0, 1])  # O(n²)
            else:
                additional_features.extend([0, 0, 0, 0, 0])
        
        # Características adicionales para Theta
        additional_features.extend([
            self._count_nested_loops(code_string) / 3.0,
            1 if self._has_recursion(code_string) else 0,
            1 if 'divide' in code_string.lower() or 'conquer' in code_string.lower() else 0
        ])
        
        return np.concatenate([base_vector, additional_features])

    def _has_early_return(self, code_string):
        """Detecta si hay retorno temprano"""
        lines = code_string.split('\n')
        for line in lines:
            if 'return' in line and 'if' in line:
                return True
        return False

    def _is_sorted_check(self, code_string):
        """Detecta verificación de ordenamiento"""
        patterns = [r'sorted', r'is_sorted', r'swapped.*=.*False']
        for pattern in patterns:
            if re.search(pattern, code_string.lower()):
                return True
        return False

    def _count_nested_loops(self, code_string):
        """Cuenta loops anidados"""
        lines = code_string.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('for ') or stripped.startswith('while '):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped == '':
                continue
            elif not stripped.startswith(' ') and current_depth > 0:
                current_depth = 0
                
        return max_depth

    def _has_recursion(self, code_string):
        """Detecta recursión"""
        import ast
        try:
            tree = ast.parse(code_string)
            func_names = set()
            
            # Encontrar definiciones de función
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_names.add(node.name)
            
            # Buscar llamadas recursivas
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in func_names:
                            return True
            return False
        except:
            return 'def ' in code_string and any(name in code_string for name in ['merge_sort', 'quick_sort'])

    def _create_specialized_model(self, model_type, input_size):
        """Crea modelos especializados según el tipo de complejidad"""
        
        if model_type == 'O':  # Big O - peor caso (más complejo)
            model = models.Sequential([
                layers.Input(shape=(input_size,)),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        elif model_type == 'Ω':  # Omega - mejor caso (más simple, enfocado en casos optimistas)
            model = models.Sequential([
                layers.Input(shape=(input_size,)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.15),
                layers.Dense(32, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        else:  # Theta - caso promedio (balance entre ambos)
            model = models.Sequential([
                layers.Input(shape=(input_size,)),
                layers.Dense(192, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                layers.Dense(96, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(48, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Configuración específica del optimizador por tipo
        if model_type == 'Ω':
            optimizer = 'adam'  # Más agresivo para mejor caso
            learning_rate = 0.001
        else:
            optimizer = 'adam'
            learning_rate = 0.0005
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def code_to_vector_enhanced(self, code_string, complexity_type='O'):
        """Convierte código a vector con características específicas por tipo"""
        try:
            base_vector = self.vectorizer.vectorize(code_string)
            if base_vector is None:
                print("Error: CodeVectorizer devolvió None")
                return None
            
            if not isinstance(base_vector, np.ndarray):
                print("Error: El vector no es un numpy array")
                return None
            
            if complexity_type == 'Ω':
                return self._get_omega_features(code_string, base_vector)
            elif complexity_type == 'Θ':
                return self._get_theta_features(code_string, base_vector)
            else:  # 'O'
                return base_vector
                
        except Exception as e:
            print(f"Error en vectorización mejorada: {str(e)}")
            return None

    def train(self, code_samples, o_labels, omega_labels, theta_labels, 
              epochs=25, batch_size=32, verbose=1):
        """Entrena los modelos con vectorización específica y callbacks mejorados"""
        print("Vectorizando código para cada tipo de complejidad...")
        
        # Vectorizar para cada tipo
        o_vectors = []
        omega_vectors = []
        theta_vectors = []
        valid_indices = []
        
        for i, code in enumerate(code_samples):
            o_vec = self.code_to_vector_enhanced(code, 'O')
            omega_vec = self.code_to_vector_enhanced(code, 'Ω')
            theta_vec = self.code_to_vector_enhanced(code, 'Θ')
            
            if all(vec is not None for vec in [o_vec, omega_vec, theta_vec]):
                o_vectors.append(o_vec)
                omega_vectors.append(omega_vec)
                theta_vectors.append(theta_vec)
                valid_indices.append(i)
            else:
                print(f"Saltando muestra {i} por error en vectorización")
        
        if not o_vectors:
            raise ValueError("No se pudieron vectorizar las muestras")
        
        # Convertir a arrays numpy
        o_vectors = np.array(o_vectors)
        omega_vectors = np.array(omega_vectors)
        theta_vectors = np.array(theta_vectors)
        
        # Filtrar etiquetas
        o_labels_filtered = np.array([o_labels[i] for i in valid_indices])
        omega_labels_filtered = np.array([omega_labels[i] for i in valid_indices])
        theta_labels_filtered = np.array([theta_labels[i] for i in valid_indices])
        
        print(f"Entrenando con {len(o_vectors)} muestras válidas de {len(code_samples)} totales")
        
        # Crear modelos con tamaños específicos
        self.o_model = self._create_specialized_model('O', o_vectors.shape[1])
        self.omega_model = self._create_specialized_model('Ω', omega_vectors.shape[1])
        self.theta_model = self._create_specialized_model('Θ', theta_vectors.shape[1])
        
        # Callbacks mejorados
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=0.00001,
            verbose=1
        )
        
        callback_list = [early_stopping, reduce_lr]
        
        # Entrenar modelos
        print("\n" + "="*50)
        print("Entrenando modelo para complejidad O (peor caso)...")
        history_o = self.o_model.fit(
            o_vectors, o_labels_filtered,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callback_list,
            verbose=verbose
        )
        
        print("\n" + "="*50)
        print("Entrenando modelo para complejidad Ω (mejor caso)...")
        history_omega = self.omega_model.fit(
            omega_vectors, omega_labels_filtered,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callback_list,
            verbose=verbose
        )
        
        print("\n" + "="*50)
        print("Entrenando modelo para complejidad Θ (caso promedio)...")
        history_theta = self.theta_model.fit(
            theta_vectors, theta_labels_filtered,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callback_list,
            verbose=verbose
        )
        
        return {
            'O': history_o,
            'Ω': history_omega,
            'Θ': history_theta
        }

    def predict_enhanced(self, code_string):
        """Predicción mejorada con lógica específica y post-procesamiento"""
        try:
            # Vectorizar para cada tipo
            o_vector = self.code_to_vector_enhanced(code_string, 'O')
            omega_vector = self.code_to_vector_enhanced(code_string, 'Ω')
            theta_vector = self.code_to_vector_enhanced(code_string, 'Θ')
            
            if any(vec is None for vec in [o_vector, omega_vector, theta_vector]):
                return None
            
            # Predicciones con probabilidades
            o_probs = self.o_model.predict(np.array([o_vector]), verbose=0)[0]
            omega_probs = self.omega_model.predict(np.array([omega_vector]), verbose=0)[0]
            theta_probs = self.theta_model.predict(np.array([theta_vector]), verbose=0)[0]
            
            o_pred = np.argmax(o_probs)
            omega_pred = np.argmax(omega_probs)
            theta_pred = np.argmax(theta_probs)
            
            # Post-procesamiento con reglas específicas
            o_result = self._complexity_class_to_string(o_pred)
            omega_result = self._post_process_omega(code_string, omega_pred)
            theta_result = self._post_process_theta(code_string, theta_pred)
            
            return {
                'O': o_result,
                'Ω': omega_result,
                'Θ': theta_result,
                'confidence': {
                    'O': float(np.max(o_probs)),
                    'Ω': float(np.max(omega_probs)),
                    'Θ': float(np.max(theta_probs))
                },
                'algorithm_detected': self._detect_algorithm_type(code_string)
            }
            
        except Exception as e:
            print(f"Error en predicción mejorada: {str(e)}")
            return None

    def _post_process_omega(self, code_string, prediction):
        """Post-procesamiento específico para Ω (mejor caso)"""
        algo_type = self._detect_algorithm_type(code_string)
        
        # Reglas específicas para mejor caso
        if algo_type == 'binary_search':
            return 'O(1)'  # Mejor caso: encontrar inmediatamente
        elif algo_type == 'linear_search':
            return 'O(1)'  # Mejor caso: encontrar en primer elemento
        elif algo_type == 'bubble_sort':
            if 'swapped' in code_string or 'break' in code_string:
                return 'O(n)'   # Bubble sort optimizado
            else:
                return 'O(n²)'  # Sin optimización
        elif algo_type == 'merge_sort':
            return 'O(n log n)'  # No cambia
        elif algo_type == 'quick_sort':
            return 'O(n log n)'  # Mejor caso con buena partición
        else:
            return self._complexity_class_to_string(prediction)

    def _post_process_theta(self, code_string, prediction):
        """Post-procesamiento específico para Θ (caso promedio)"""
        algo_type = self._detect_algorithm_type(code_string)
        
        # Correcciones específicas para caso promedio
        if algo_type == 'binary_search':
            return 'O(log n)'  # Caso promedio
        elif algo_type == 'linear_search':
            return 'O(n)'      # Caso promedio
        elif algo_type == 'bubble_sort':
            return 'O(n²)'     # Caso promedio
        elif algo_type == 'merge_sort':
            return 'O(n log n)' # Siempre igual
        elif algo_type == 'quick_sort':
            return 'O(n log n)' # Caso promedio
        else:
            return self._complexity_class_to_string(prediction)

    def predict(self, code_string):
        """Función de predicción compatible con versión anterior"""
        result = self.predict_enhanced(code_string)
        if result:
            return {
                'O': result['O'],
                'Ω': result['Ω'],
                'Θ': result['Θ'],
                'confidence': result['confidence']
            }
        return None

    def _complexity_class_to_string(self, class_index):
        """Mapea índices a strings de complejidad"""
        return self.complexity_map.get(class_index, f"Clase_{class_index}")

    def evaluate(self, test_code_samples, test_o_labels, test_omega_labels, test_theta_labels):
        """Evaluación mejorada con análisis por algoritmo"""
        o_vectors = []
        omega_vectors = []
        theta_vectors = []
        valid_indices = []
        algorithm_types = []
        
        # Vectorizar muestras de prueba
        for i, code in enumerate(test_code_samples):
            o_vec = self.code_to_vector_enhanced(code, 'O')
            omega_vec = self.code_to_vector_enhanced(code, 'Ω')
            theta_vec = self.code_to_vector_enhanced(code, 'Θ')
            
            if all(vec is not None for vec in [o_vec, omega_vec, theta_vec]):
                o_vectors.append(o_vec)
                omega_vectors.append(omega_vec)
                theta_vectors.append(theta_vec)
                valid_indices.append(i)
                algorithm_types.append(self._detect_algorithm_type(code))
        
        if not o_vectors:
            print("Error: No se pudieron vectorizar las muestras de prueba")
            return None
        
        # Convertir a arrays
        o_vectors = np.array(o_vectors)
        omega_vectors = np.array(omega_vectors)
        theta_vectors = np.array(theta_vectors)
        
        # Filtrar etiquetas
        o_labels_filtered = np.array([test_o_labels[i] for i in valid_indices])
        omega_labels_filtered = np.array([test_omega_labels[i] for i in valid_indices])
        theta_labels_filtered = np.array([test_theta_labels[i] for i in valid_indices])
        
        # Evaluaciones básicas
        o_eval = self.o_model.evaluate(o_vectors, o_labels_filtered, verbose=0)
        omega_eval = self.omega_model.evaluate(omega_vectors, omega_labels_filtered, verbose=0)
        theta_eval = self.theta_model.evaluate(theta_vectors, theta_labels_filtered, verbose=0)
        
        # Predicciones
        o_pred = np.argmax(self.o_model.predict(o_vectors, verbose=0), axis=1)
        omega_pred = np.argmax(self.omega_model.predict(omega_vectors, verbose=0), axis=1)
        theta_pred = np.argmax(self.theta_model.predict(theta_vectors, verbose=0), axis=1)
        
        return {
            'O': {
                'loss': o_eval[0], 
                'accuracy': o_eval[1],
                'predictions': o_pred,
                'true_labels': o_labels_filtered
            },
            'Ω': {
                'loss': omega_eval[0], 
                'accuracy': omega_eval[1],
                'predictions': omega_pred,
                'true_labels': omega_labels_filtered
            },
            'Θ': {
                'loss': theta_eval[0], 
                'accuracy': theta_eval[1],
                'predictions': theta_pred,
                'true_labels': theta_labels_filtered
            },
            'algorithm_types': algorithm_types,
            'valid_indices': valid_indices
        }

    def print_detailed_evaluation(self, evaluation_results):
        """Imprime evaluación detallada con matrices de confusión y análisis por algoritmo"""
        complexity_types = ['O', 'Ω', 'Θ']
        
        for comp_type in complexity_types:
            if comp_type in evaluation_results:
                result = evaluation_results[comp_type]
                print(f"\n{'='*60}")
                print(f"EVALUACIÓN DETALLADA PARA {comp_type}")
                print(f"{'='*60}")
                print(f"Pérdida: {result['loss']:.4f}")
                print(f"Precisión: {result['accuracy']:.4f}")
                
                if 'predictions' in result and 'true_labels' in result:
                    print(f"\n{'-'*40}")
                    print("REPORTE DE CLASIFICACIÓN:")
                    print(f"{'-'*40}")
                    target_names = [self._complexity_class_to_string(i) for i in range(self.num_classes)]
                    print(classification_report(
                        result['true_labels'], 
                        result['predictions'],
                        target_names=target_names,
                        zero_division=0
                    ))
                    
                    print(f"\n{'-'*40}")
                    print("MATRIZ DE CONFUSIÓN:")
                    print(f"{'-'*40}")
                    cm = confusion_matrix(result['true_labels'], result['predictions'])
                    print(f"Etiquetas: {target_names}")
                    print(cm)
        
        # Análisis por tipo de algoritmo si está disponible
        if 'algorithm_types' in evaluation_results:
            print(f"\n{'='*60}")
            print("ANÁLISIS POR TIPO DE ALGORITMO")
            print(f"{'='*60}")
            
            algo_counts = {}
            for algo in evaluation_results['algorithm_types']:
                algo_counts[algo] = algo_counts.get(algo, 0) + 1
            
            for algo, count in algo_counts.items():
                print(f"{algo}: {count} muestras")

    def save_models(self, base_path="complexity_models"):
        """Guarda los modelos entrenados"""
        if self.o_model:
            self.o_model.save(f"{base_path}_O.h5")
        if self.omega_model:
            self.omega_model.save(f"{base_path}_Omega.h5")
        if self.theta_model:
            self.theta_model.save(f"{base_path}_Theta.h5")
        print(f"Modelos guardados en {base_path}_*.h5")

    def load_models(self, base_path="complexity_models"):
        """Carga los modelos guardados"""
        try:
            self.o_model = models.load_model(f"{base_path}_O.h5")
            self.omega_model = models.load_model(f"{base_path}_Omega.h5")
            self.theta_model = models.load_model(f"{base_path}_Theta.h5")
            print(f"Modelos cargados desde {base_path}_*.h5")
            return True
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            return False