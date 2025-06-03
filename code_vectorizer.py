import ast
import numpy as np
from collections import defaultdict

class CodeVectorizer:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.node_types = [
            ast.FunctionDef, ast.For, ast.While, ast.If, ast.Call, ast.BinOp,
            ast.Compare, ast.Assign, ast.Return, ast.Name, ast.Num, ast.Str,
            ast.List, ast.Dict, ast.Subscript, ast.Attribute
        ]
        
        # Parámetros de normalización mejorados
        self.max_nested_loops = 5
        self.max_loop_variables = 10
        self.max_loop_operations = 20
        self.max_recursive_calls = 10
        self.max_index_access = 15
        self.max_tree_depth = 15
        self.max_nested_ifs = 8
        self.max_total_calls = 20

    def _count_node_types(self, node):
        """Cuenta tipos de nodos AST."""
        counts = {node_type.__name__: 0 for node_type in self.node_types}

        def visit(n):
            for node_type in self.node_types:
                if isinstance(n, node_type):
                    counts[node_type.__name__] += 1
            for child in ast.iter_child_nodes(n):
                visit(child)

        visit(node)
        return counts

    def _extract_loop_features(self, node):
        """
        Extrae características de loops MEJORADO - no ignora loops en funciones recursivas.
        """
        features = {
            'nested_loops': 0, 
            'loop_variables': set(), 
            'loop_operations': 0,
            'max_loop_depth': 0,
            'total_loops': 0
        }

        def visit(n, loop_depth=0):
            current_max_depth = loop_depth
            
            if isinstance(n, (ast.For, ast.While)):
                features['total_loops'] += 1
                current_loop_depth = loop_depth + 1
                features['max_loop_depth'] = max(features['max_loop_depth'], current_loop_depth)
                
                # Contar variables de loop
                if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
                    features['loop_variables'].add(n.target.id)
                
                # CORREGIDO: Contar operaciones DENTRO del loop independientemente de recursión
                def count_operations_in_loop(loop_node):
                    ops = 0
                    for child in ast.walk(loop_node):
                        if isinstance(child, (ast.BinOp, ast.Compare, ast.Call, ast.Subscript)):
                            ops += 1
                    return ops
                
                features['loop_operations'] += count_operations_in_loop(n)
                
                # Continuar con hijos
                for child in ast.iter_child_nodes(n):
                    visit(child, current_loop_depth)
            else:
                # Para nodos no-loop, mantener la profundidad actual
                for child in ast.iter_child_nodes(n):
                    visit(child, loop_depth)
        
        visit(node)
        features['nested_loops'] = features['max_loop_depth']
        return features

    def _extract_recursion_features(self, node):
        """
        Extrae características de recursión MEJORADO - detecta patrones de recursión.
        """
        features = {
            'has_recursion': False, 
            'recursive_calls': 0,
            'recursion_pattern': 'none',  # 'linear', 'binary', 'multiple'
            'recursive_functions': set()
        }

        function_definitions = {}
        
        # Primera pasada: encontrar todas las definiciones de función
        for node_item in ast.walk(node):
            if isinstance(node_item, ast.FunctionDef):
                function_definitions[node_item.name] = node_item

        # Segunda pasada: analizar calls recursivos
        def analyze_function(func_name, func_node):
            if func_name in features['recursive_functions']:
                return  # Ya analizado
                
            recursive_calls_in_func = 0
            calls_pattern = []
            
            for item in ast.walk(func_node):
                if isinstance(item, ast.Call):
                    call_name = None
                    if isinstance(item.func, ast.Name):
                        call_name = item.func.id
                    elif isinstance(item.func, ast.Attribute):
                        call_name = item.func.attr
                    
                    if call_name == func_name:
                        recursive_calls_in_func += 1
                        calls_pattern.append(item)
            
            if recursive_calls_in_func > 0:
                features['has_recursion'] = True
                features['recursive_calls'] += recursive_calls_in_func
                features['recursive_functions'].add(func_name)
                
                # Determinar patrón de recursión
                if recursive_calls_in_func == 1:
                    features['recursion_pattern'] = 'linear'
                elif recursive_calls_in_func == 2:
                    features['recursion_pattern'] = 'binary'  # Como merge sort
                else:
                    features['recursion_pattern'] = 'multiple'

        # Analizar cada función
        for func_name, func_node in function_definitions.items():
            analyze_function(func_name, func_node)

        return features

    def _extract_divide_conquer_features(self, node):
        """
        NUEVO: Detecta patrones específicos de divide y vencerás.
        """
        features = {
            'has_divide_conquer': False,
            'array_slicing': 0,
            'midpoint_calculation': 0,
            'merge_pattern': False
        }
        
        # Buscar patrones típicos de divide y vencerás
        for item in ast.walk(node):
            # Detectar cálculo de punto medio: mid = (left + right) // 2 o similar
            if isinstance(item, ast.Assign):
                if isinstance(item.value, ast.BinOp):
                    if isinstance(item.value.op, ast.FloorDiv):
                        features['midpoint_calculation'] += 1
            
            # Detectar slicing de arrays: arr[:mid], arr[mid:]
            elif isinstance(item, ast.Subscript):
                if isinstance(item.slice, ast.Slice):
                    features['array_slicing'] += 1
            
            # Detectar patrones de merge
            elif isinstance(item, ast.Call):
                if isinstance(item.func, ast.Name):
                    if 'merge' in item.func.id.lower():
                        features['merge_pattern'] = True

        # Si tiene recursión + slicing + merge = divide y vencerás
        if features['array_slicing'] >= 2 and features['merge_pattern']:
            features['has_divide_conquer'] = True
            
        return features

    def _detect_index_access(self, node):
        """Detecta accesos a índices."""
        count = 0
        class Visitor(ast.NodeVisitor):
            def visit_Subscript(self, sub_node):
                nonlocal count
                count += 1
                self.generic_visit(sub_node)

        Visitor().visit(node)
        return count

    def _calculate_tree_depth(self, node):
        """Calcula profundidad del árbol AST."""
        if not list(ast.iter_child_nodes(node)):
            return 1
        return 1 + max(self._calculate_tree_depth(child) for child in ast.iter_child_nodes(node))

    def _count_nested_ifs(self, node):
        """Cuenta profundidad máxima de ifs anidados."""
        max_depth = 0

        def visit(n, depth=0):
            nonlocal max_depth
            if isinstance(n, ast.If):
                max_depth = max(max_depth, depth + 1)
                for child in ast.iter_child_nodes(n):
                    visit(child, depth + 1)
            else:
                for child in ast.iter_child_nodes(n):
                    visit(child, depth)

        visit(node)
        return max_depth

    def _count_function_calls(self, node):
        """Cuenta llamadas a funciones."""
        counts = defaultdict(int)

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, call_node):
                func_name = None
                if isinstance(call_node.func, ast.Name):
                    func_name = call_node.func.id
                elif isinstance(call_node.func, ast.Attribute):
                    func_name = call_node.func.attr
                if func_name:
                    counts[func_name] += 1
                self.generic_visit(call_node)

        Visitor().visit(node)
        return counts

    def _extract_complexity_hints(self, node):
        """
        NUEVO: Extrae pistas específicas de complejidad algorítmica.
        """
        features = {
            'binary_search_pattern': False,
            'nested_loop_pattern': False,
            'single_loop_pattern': False,
            'constant_time_pattern': False,
            'logarithmic_division': False
        }
        
        loop_features = self._extract_loop_features(node)
        recursion_features = self._extract_recursion_features(node)
        
        # Patrón de búsqueda binaria
        if (not recursion_features['has_recursion'] and 
            loop_features['total_loops'] == 1 and 
            'left' in str(node) and 'right' in str(node) and 'mid' in str(node)):
            features['binary_search_pattern'] = True
        
        # Patrón de loops anidados
        if loop_features['nested_loops'] >= 2:
            features['nested_loop_pattern'] = True
        elif loop_features['nested_loops'] == 1:
            features['single_loop_pattern'] = True
        
        # Patrón de tiempo constante
        if (loop_features['total_loops'] == 0 and 
            not recursion_features['has_recursion']):
            features['constant_time_pattern'] = True
        
        # División logarítmica (divide y vencerás)
        if (recursion_features['recursion_pattern'] == 'binary' and
            'mid' in str(node)):
            features['logarithmic_division'] = True
            
        return features

    def vectorize(self, code_string):
        """
        Vectoriza código MEJORADO con características específicas de complejidad.
        """
        if not isinstance(code_string, str):
            print("Error: La entrada debe ser una cadena de texto.")
            return np.zeros(self.vector_size)

        if not code_string.strip():
            print("Error: El código está vacío.")
            return np.zeros(self.vector_size)

        try:
            tree = ast.parse(code_string)

            # Extraer todas las características
            node_counts = self._count_node_types(tree)
            loop_features = self._extract_loop_features(tree)
            recursion_features = self._extract_recursion_features(tree)
            divide_conquer_features = self._extract_divide_conquer_features(tree)
            complexity_hints = self._extract_complexity_hints(tree)
            
            index_access = self._detect_index_access(tree)
            tree_depth = self._calculate_tree_depth(tree)
            nested_ifs = self._count_nested_ifs(tree)
            function_calls = self._count_function_calls(tree)

            features = []
            total_nodes = sum(node_counts.values())

            # Características de nodos AST (normalizadas)
            for node_type in self.node_types:
                count = node_counts[node_type.__name__]
                features.append(count / total_nodes if total_nodes > 0 else 0)

            # Características de loops (CORREGIDAS)
            features.extend([
                loop_features['nested_loops'] / self.max_nested_loops,
                len(loop_features['loop_variables']) / self.max_loop_variables,
                loop_features['loop_operations'] / self.max_loop_operations,
                loop_features['total_loops'] / 10.0  # Nueva característica
            ])

            # Características de recursión (MEJORADAS)
            features.extend([
                1 if recursion_features['has_recursion'] else 0,
                recursion_features['recursive_calls'] / self.max_recursive_calls,
                1 if recursion_features['recursion_pattern'] == 'linear' else 0,
                1 if recursion_features['recursion_pattern'] == 'binary' else 0,
                1 if recursion_features['recursion_pattern'] == 'multiple' else 0
            ])

            # Características de divide y vencerás (NUEVAS)
            features.extend([
                1 if divide_conquer_features['has_divide_conquer'] else 0,
                divide_conquer_features['array_slicing'] / 10.0,
                divide_conquer_features['midpoint_calculation'] / 5.0,
                1 if divide_conquer_features['merge_pattern'] else 0
            ])

            # Pistas de complejidad (NUEVAS)
            features.extend([
                1 if complexity_hints['binary_search_pattern'] else 0,
                1 if complexity_hints['nested_loop_pattern'] else 0,
                1 if complexity_hints['single_loop_pattern'] else 0,
                1 if complexity_hints['constant_time_pattern'] else 0,
                1 if complexity_hints['logarithmic_division'] else 0
            ])

            # Características adicionales
            features.extend([
                index_access / self.max_index_access,
                tree_depth / self.max_tree_depth,
                nested_ifs / self.max_nested_ifs,
                sum(function_calls.values()) / self.max_total_calls
            ])

            # Ajustar tamaño del vector
            features = np.array(features)
            if len(features) < self.vector_size:
                features = np.pad(features, (0, self.vector_size - len(features)))
            else:
                features = features[:self.vector_size]

            return features

        except Exception as e:
            print(f"Error al vectorizar el código: {e}")
            return np.zeros(self.vector_size)