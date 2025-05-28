import ast
import numpy as np
from collections import defaultdict

class CodeVectorizer:
    def __init__(self, vector_size=101):  # Aumentado a 101 para incluir nueva característica
        self.vector_size = vector_size
        self.node_types = [
            ast.For, ast.While, ast.If, ast.FunctionDef,
            ast.Call, ast.Compare, ast.BinOp, ast.Return,
            ast.Assign, ast.Name, ast.Num, ast.List
        ]
        self.feature_count = defaultdict(int)

    def _count_node_types(self, node):
        """Cuenta los diferentes tipos de nodos en el AST"""
        counts = defaultdict(int)
        for node_type in self.node_types:
            counts[node_type.__name__] = 0

        def visit(node):
            for node_type in self.node_types:
                if isinstance(node, node_type):
                    counts[node_type.__name__] += 1
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(node)
        return counts

    def _extract_loop_features(self, node):
        """Extrae características específicas de los bucles"""
        features = {
            'nested_loops': 0,
            'loop_variables': set(),
            'loop_operations': 0
        }

        def visit(node, depth=0):
            if isinstance(node, (ast.For, ast.While)):
                features['nested_loops'] = max(features['nested_loops'], depth + 1)
                
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                    features['loop_variables'].add(node.target.id)

                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.BinOp, ast.Compare, ast.Call)):
                        features['loop_operations'] += 1
                    visit(child, depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    visit(child, depth)

        visit(node)
        return features

    def _extract_recursion_features(self, node, function_name=None):
        """Detecta y analiza patrones de recursión"""
        features = {
            'has_recursion': False,
            'recursion_depth': 0,
            'recursive_calls': 0
        }

        def visit(node, current_function=None):
            if isinstance(node, ast.FunctionDef):
                current_function = node.name

            if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                if node.func.id == current_function:
                    features['has_recursion'] = True
                    features['recursive_calls'] += 1

            for child in ast.iter_child_nodes(node):
                visit(child, current_function)

        visit(node)
        return features

    def _detect_index_access(self, node):
        """Detecta accesos directos tipo arr[0]"""
        index_access_count = 0

        class IndexAccessVisitor(ast.NodeVisitor):
            def visit_Subscript(self, sub_node):
                nonlocal index_access_count
                index_access_count += 1
                self.generic_visit(sub_node)

        IndexAccessVisitor().visit(node)
        return index_access_count


    def vectorize(self, code_string):
        """Convierte el código en un vector de características"""
        if not isinstance(code_string, str):
            print("Error: La entrada debe ser una cadena de texto")
            return None
            
        if not code_string.strip():
            print("Error: El código está vacío")
            return None

        try:
            tree = ast.parse(code_string)
            
            # Extraer características básicas
            node_counts = self._count_node_types(tree)
            loop_features = self._extract_loop_features(tree)
            recursion_features = self._extract_recursion_features(tree)
            index_access = self._detect_index_access(tree)

            # Crear vector de características
            features = []
            
            # Agregar conteos de nodos normalizados
            total_nodes = sum(node_counts.values())
            for node_type in self.node_types:
                if total_nodes > 0:
                    features.append(node_counts[node_type.__name__] / total_nodes)
                else:
                    features.append(0)

            # Agregar características de bucles
            features.extend([
                loop_features['nested_loops'] / 5,
                len(loop_features['loop_variables']) / 10,
                loop_features['loop_operations'] / 20
            ])

            # Agregar características de recursión
            features.extend([
                1 if recursion_features['has_recursion'] else 0,
                recursion_features['recursive_calls'] / 5
            ])

            features.append(index_access / 10)

            # Asegurar dimensión consistente
            features = np.array(features)
            if len(features) < self.vector_size:
                features = np.pad(features, (0, self.vector_size - len(features)))
            else:
                features = features[:self.vector_size]

            return features

        except Exception as e:
            print(f"Error al vectorizar el código: {e}")
            return np.zeros(self.vector_size)

