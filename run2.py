import sys
from collections import deque, defaultdict


def solve(edges: list[tuple[str, str]]) -> list[str]:
    """
    Решение задачи об изоляции вируса

    Args:
        edges: список коридоров в формате (узел1, узел2)

    Returns:
        список отключаемых коридоров в формате "Шлюз-узел"
    """
    # Строим граф
    graph = defaultdict(list)
    gateways = set()
    nodes = set()

    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        nodes.add(u)
        nodes.add(v)
        # Определяем шлюзы
        if u.isupper():
            gateways.add(u)
        if v.isupper():
            gateways.add(v)

    virus_pos = 'a'
    result = []

    # Алгоритм решения
    while True:
        # Находим все коридоры для отключения
        available_cuts = []
        for gateway in gateways:
            for neighbor in graph[gateway]:
                if neighbor.islower():
                    available_cuts.append((gateway, neighbor))

        # Сортируем по лексикографическому порядку и по паре - заглавная и строчная буквы
        available_cuts.sort(key=lambda x: (x[0], x[1]))

        # Находим ближайший для вируса шлюз
        target_gateway = find_target_gateway(graph, virus_pos, gateways)

        if target_gateway is None:
            break

        # Находим следующий шаг для вируса
        next_node = find_next_move(graph, virus_pos, target_gateway)

        best_cut = None
        # Выбираем коридор, который собираемся отключить
        for gateway, node in available_cuts:
            if would_block_path(graph, virus_pos, target_gateway, (gateway, node)):
                best_cut = (gateway, node)
                break

        # Если не нашли подходящего отключения, то берем первый доступный в лексикографическом порядке
        if best_cut is None and available_cuts:
            best_cut = available_cuts[0]

        if best_cut is None:
            break

        # Результаты добавляем в result
        gateway, node = best_cut
        result.append(f"{gateway}-{node}")

        # Удаляем коридор из графа
        graph[gateway].remove(node)
        graph[node].remove(gateway)

        # Вирус перемещается
        if next_node is not None:
            virus_pos = next_node
        else:
            # Если вирус не может передвигаться - завершаем
            break

    return result

# Находим ближайший для вируса шлюз
def find_target_gateway(graph, start, gateways):
    distances = {}

    for gateway in gateways:
        dist = breadth_distance(graph, start, gateway)
        if dist is not None:
            distances[gateway] = dist

    if not distances:
        return None

    # Находим наименьшее расстояние
    min_dist = min(distances.values())

    # Среди шлюзов с наименьшим расстоянием выбираем по лексикографическому порядку
    candidates = [way for way, dist in distances.items() if dist == min_dist]
    candidates.sort()

    return candidates[0] if candidates else None

# Находим следующий шаг вируса к ближайшему шлюзу
def find_next_move(graph, current_pos, target_gateway):
    # Используем поиск в ширину для нахождения всех кратчайших путей
    queue = deque([(current_pos, [current_pos])])
    visited = {current_pos}
    shortest_paths = []
    min_length = float('inf')

    while queue:
        node, path = queue.popleft()

        if len(path) > min_length:
            continue

        if node == target_gateway:
            if len(path) < min_length:
                min_length = len(path)
                shortest_paths = [path]
            elif len(path) == min_length:
                shortest_paths.append(path)
            continue

        # Сортируем для требования детерминированности
        for neighbor in sorted(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    if not shortest_paths:
        return None

    # Из кратчайших путей выбираем следующим шагом наименьший по лексикографическому порядку
    next_nodes = set()
    for path in shortest_paths:
        if len(path) > 1:
            next_nodes.add(path[1])

    return min(next_nodes) if next_nodes else None

# Проверяем блокирует ли отключение коридора путь к ближайшему для вируса шлюзу
def would_block_path(graph, start, target, cut_edge):
    gateway, node = cut_edge

    # Удаляем коридор
    graph[gateway].remove(node)
    graph[node].remove(gateway)

    # Проверяем путь
    has_path = breadth_distance(graph, start, target) is not None

    # Возвращаем коридор
    graph[gateway].append(node)
    graph[node].append(gateway)

    return not has_path

# Ищем кратчайшее расстояние между узлами поиском в ширину
def breadth_distance(graph, start, end):
    if start == end:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, dist = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return None


def main():
    edges = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))

    result = solve(edges)
    for edge in result:
        print(edge)


if __name__ == "__main__":
    main()