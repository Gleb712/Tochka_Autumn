import sys
import heapq
from typing import List, Tuple, Iterable

# Стоимость перемещения по типам
ENERGY = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}

# Позиции входов в комнаты на линии коридора
ROOM_HALL_POS = (2, 4, 6, 8)

# места, где нельзя останавливаться
HALL_FORBIDDEN = set(ROOM_HALL_POS)

# Разрешенные для остановки позиции
HALL_POSITIONS = tuple(i for i in range(11) if i not in HALL_FORBIDDEN)


# Разбор входных данных
def parse_input(lines: List[str]):
    # коридор
    hallway = tuple(lines[1][1:12])

    # колонки, где находятся комнаты
    cols = [3, 5, 7, 9]

    # строки с комнатами
    room_rows = lines[2:-1]

    depth = len(room_rows)
    rooms = [tuple(row[c] for row in room_rows) for c in cols]
    return tuple(hallway), tuple(rooms)


# Целевое состояние
def rooms_goal(depth: int):
    return tuple(tuple(ch for _ in range(depth)) for ch in "ABCD")


# Проверка завершения
def is_finished(rooms: Tuple[Tuple[str, ...], ...]):
    for i, room in enumerate(rooms):
        if any(c != "ABCD"[i] for c in room):
            return False
    return True


# Проверка, что путь в коридоре свободен
def path_clear(hallway: Tuple[str, ...], start: int, end: int):
    if start == end:
        return True
    if start < end:
        segment = hallway[start + 1:end + 1]
    else:
        segment = hallway[end:start]
    return all(c == '.' for c in segment)


# Ходы из коридора в комнаты
def moves_from_hallway_to_rooms(hallway: Tuple[str, ...], rooms: Tuple[Tuple[str, ...], ...]):
    depth = len(rooms[0])
    for pos, ch in enumerate(hallway):
        if ch == '.':
            continue
        room_idx = ord(ch) - ord('A')
        room_pos = ROOM_HALL_POS[room_idx]
        if not path_clear(hallway, pos, room_pos):
            continue
        room = rooms[room_idx]
        if any(c != '.' and c != ch for c in room):
            continue
        # Вставляем в самую глубокую свободную позицию
        for d in range(depth - 1, -1, -1):
            if room[d] == '.':
                new_hall = list(hallway)
                new_rooms = [list(r) for r in rooms]
                new_hall[pos] = '.'
                new_rooms[room_idx][d] = ch
                steps = abs(pos - room_pos) + (d + 1)
                cost = steps * ENERGY[ch]
                yield (tuple(new_hall), tuple(tuple(r) for r in new_rooms)), cost
                break


# Ходы из комнат в коридор
def moves_from_rooms_to_hallway(
    hallway: Tuple[str, ...], rooms: Tuple[Tuple[str, ...], ...]
):
    depth = len(rooms[0])
    for i, room in enumerate(rooms):
        room_ch = "ABCD"[i]
        if all(c == '.' for c in room):
            continue
        # Не двигаем, если все уже на своих местах
        if all(c == '.' or c == room_ch for c in room):
            continue

        top_idx = next((j for j, c in enumerate(room) if c != '.'), None)
        ch = room[top_idx]
        start_pos = ROOM_HALL_POS[i]

        for target in HALL_POSITIONS:
            if not path_clear(hallway, start_pos, target):
                continue
            new_hall = list(hallway)
            new_rooms = [list(r) for r in rooms]
            new_hall[target] = ch
            new_rooms[i][top_idx] = '.'
            steps = abs(target - start_pos) + (top_idx + 1)
            cost = steps * ENERGY[ch]
            yield (tuple(new_hall), tuple(tuple(r) for r in new_rooms)), cost


def solve(lines: List[str]) -> int:
    """
        Решение задачи о сортировке в лабиринте

        Args:
            lines: список строк, представляющих лабиринт

        Returns:
            минимальная энергия для достижения целевой конфигурации
    """

    hallway, rooms = parse_input(lines)
    depth = len(rooms[0])
    goal = rooms_goal(depth)
    start = (hallway, rooms)

    heap = [(0, start)]
    best = {start: 0}

    while heap:
        cost, state = heapq.heappop(heap)
        if best.get(state, 10**9) != cost:
            continue
        hall, rms = state
        if rms == goal:
            return cost

        for next_state, move_cost in moves_from_hallway_to_rooms(hall, rms):
            new_cost = cost + move_cost
            if new_cost < best.get(next_state, 10**9):
                best[next_state] = new_cost
                heapq.heappush(heap, (new_cost, next_state))

        for next_state, move_cost in moves_from_rooms_to_hallway(hall, rms):
            new_cost = cost + move_cost
            if new_cost < best.get(next_state, 10**9):
                best[next_state] = new_cost
                heapq.heappush(heap, (new_cost, next_state))

    return 0


def main():
    # Чтение входных данных
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()
