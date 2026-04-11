#!/usr/bin/env python3
import ollama
import subprocess
import sys
import re

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGRetriever:
    """Локальный поиск похожих примеров из файла с тестами"""
    
    def __init__(self, test_file_path="test_examples.json"):
        self.test_file_path = test_file_path
        self.chunks = []          # список чанков (текст задачи + код)
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Загружаем и индексируем файл
        self._load_and_index()
    
    def _load_and_index(self):
        """Загружает тестовые задания и строит TF-IDF индекс"""
        if not Path(self.test_file_path).exists():
            print(f"⚠️ Файл {self.test_file_path} не найден, RAG отключён")
            return
        
        # Загружаем JSON (предполагаем структуру как в примере)
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Извлекаем все пары (запрос, ожидаемый код)
        # Адаптируйте под структуру вашего файла
        for item in data.get("examples", []):
            query = item.get("query", "")
            code = item.get("expected_code", "")
            if query and code:
                self.chunks.append({
                    "query": query,
                    "code": code,
                    "text": query + " " + code  # для поиска
                })
        
        # Если нет структуры, можно распарсить текстовый файл
        # (альтернативный метод ниже)
        if not self.chunks:
            self._parse_text_file()
        
        # Строим TF-IDF матрицу (легковесно, без внешних API)
        if self.chunks:
            texts = [chunk["text"] for chunk in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words=None, max_features=500)
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print(f"✅ Загружено {len(self.chunks)} примеров для RAG")
    
    def _parse_text_file(self):
        """Если файл в текстовом формате (как в примере) — извлекаем пары"""
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем блоки "Запрос пользователя" и "Ожидаемый ответ"
        # Это эвристика под ваш файл
        pattern = r"Запрос пользователя\s*\n(.*?)\n.*?Ожидаемый ответ\s*\n(.*?)(?=\n\w|$)"
        matches = re.findall(pattern, content, re.DOTALL)
        
        for query, code in matches:
            query = query.strip()
            code = code.strip()
            if query and code:
                self.chunks.append({
                    "query": query,
                    "code": code,
                    "text": query + " " + code
                })
    
    def retrieve(self, user_query, top_k=2):
        """Возвращает top_k самых похожих примеров"""
        if not self.chunks or self.vectorizer is None:
            return []
        
        # Векторизуем запрос пользователя
        query_vec = self.vectorizer.transform([user_query])
        
        # Считаем косинусное сходство
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Берём top_k индексов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # порог релевантности
                results.append(self.chunks[idx])
        
        return results

class LuaCodeAgent:
    def __init__(self, model_name="deepseek-coder:6.7b-instruct-q4_K_M", test_file="test_examples.json"):
        self.model_name = model_name
        self.conversation_history = []
        self.max_iterations = 3
        self.rag = RAGRetriever(test_file)
        self.system_prompt = """You are an expert Lua 5.5 developer with 10 years of experience. Your code is used in production systems.

CRITICAL RULES:
1. ALWAYS handle edge cases (nil, empty tables, wrong types)
2. ALWAYS add input validation at the start of functions
3. ALWAYS include comments explaining complex logic
4. ALWAYS write tests using assert() after the function
5. NEVER use global variables unless specified
6. NEVER write code that can crash (protect against stack overflow, infinite loops)

CODE QUALITY STANDARDS:
- Use descriptive variable names
- Add type hints in comments (--- @param, --- @return)
- Prefer ipairs() for arrays, pairs() for dictionaries
- Check types with type() before operations
- Return empty table {} instead of nil for missing data

OUTPUT FORMAT:
```lua
--- Function description
--- @param param_name type description
--- @return type description
function name(param)
    -- validation
    -- logic
    -- return
end

MATHEMATICAL FUNCTIONS RULES:
- For power function (x^n): 
  * Negative exponents: result = 1 / (x^positive)
  * x^0 = 1 for any x ≠ 0
  * 0^0 is undefined (error)
  * 0^positive = 0
  * 0^negative = error (division by zero)
  * Even exponent of negative number = positive result
  * Odd exponent of negative number = negative result
- Always handle integer exponents, use fast exponentiation
- Use math.abs() for floating point comparisons in tests
- Never write incorrect test assertions

CRITICAL RULE FOR POWER FUNCTIONS:
NEVER implement power(x,n) yourself. ALWAYS use math.pow(x, n).

Example of CORRECT code:
function power(x, n)
    if type(x) ~= "number" or type(n) ~= "number" then
        error("Invalid input")
    end
    return math.pow(x, n)
end

-- TESTS
local test1 = ...
assert(test1 == expected, "Test failed: description")
REMEMBER: Production-ready code only. No placeholders. No "TODO" comments."""
    

    def _call_llm(self, prompt, role="user"):
        if not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": self.system_prompt})
        self.conversation_history.append({"role": role, "content": prompt})
        if len(self.conversation_history) > 15:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-14:]

        response = ollama.chat(
            model=self.model_name,
            messages=self.conversation_history,
        )
        return response['message']['content']

    def _extract_code(self, llm_output):
        """Extracts Lua code from LLM response."""
        code_blocks = re.findall(r'```lua\n(.*?)```', llm_output, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # Если нет маркера lua, ищем любой блок кода
        code_blocks = re.findall(r'```\n(.*?)```', llm_output, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # Если нет блоков, возможно весь ответ - это код
        return llm_output.strip()

    def _validate_code(self, code):
        """Проверяет синтаксис Lua кода через luac"""
        # Сохраняем код во временный файл
        with open('/tmp/test_script.lua', 'w') as f:
            f.write(code)

        # Проверяем синтаксис через luac
        result = subprocess.run(['luac', '-p', '/tmp/test_script.lua'], capture_output=True, text=True)

        if result.returncode != 0:
            return False, f"Syntax error: {result.stderr}"

        return True, "Code is valid"

    def run(self, user_query):
        """Главный цикл агента."""
        print(f"🤖 Получен запрос: {user_query}")
        self.conversation_history = []  # Сбрасываем историю для нового запроса

        # 1. Ищем похожие примеры из тестового файла
        similar_examples = self.rag.retrieve(user_query, top_k=2)
    
        # 2. Формируем контекст
        rag_context = ""
        if similar_examples:
            rag_context = "\n\n--- ПРИМЕРЫ ИЗ ТЕСТОВОГО ФАЙЛА ---\n"
            for i, ex in enumerate(similar_examples, 1):
              rag_context += f"\nПример {i}:\nЗапрос: {ex['query']}\nКод:\n```lua\n{ex['code']}\n```\n"
            rag_context += "\n--- ИСПОЛЬЗУЙ ЭТИ ПРИМЕРЫ КАК РЕФЕРЕНС ---\n"
            print(f"📚 Найдено {len(similar_examples)} похожих примеров")
    
        # 3. Добавляем контекст в промпт
        enhanced_prompt = f"{rag_context}\n\nЗадача: {user_query}"


        # Шаг 1: Планирование
        plan_prompt = f"Create a plan to write Lua code for the following task: {user_query}"
        plan = self._call_llm(plan_prompt)
        print(f"📝 План: {plan[:200]}...")

        # Шаг 2: Генерация и итеративное улучшение
        current_code = ""
        last_validation_feedback = ""

        for i in range(self.max_iterations):
            print(f"🔄 Итерация {i+1}: Генерация/Улучшение кода...")

            # Генерация (или рефакторинг)
            if i == 0:
                code_prompt = f"""Generate production-ready Lua code for this task:

TASK: {user_query}

REQUIREMENTS:
- Write complete, runnable code
- Include input validation
- Handle edge cases
- Add 3-5 test cases with assert()
- Return ONLY the code in ```lua block

Follow the system instructions strictly."""
            else:
                code_prompt = f"""Fix this Lua code to pass validation:

ERRORS TO FIX:
{last_validation_feedback}

CURRENT CODE:
```lua
{current_code}
```
FIX:

Add missing validations
Fix syntax errors
Add/update tests
Return complete fixed code in ```lua block"""

            llm_output = self._call_llm(code_prompt)
            current_code = self._extract_code(llm_output)

            if not current_code:
                print("  ❌ Не удалось извлечь код")
                continue

            print(f"  📄 Сгенерирован код ({len(current_code)} символов)")

            # Шаг 3: Валидация
            is_valid, validation_feedback = self._validate_code(current_code)
            print(f"🔍 Результат валидации: {'✅ Успешно' if is_valid else '❌ Ошибка'}")

            if is_valid:
                print("✅ Код успешно прошел проверку!")
                return current_code, True, self.conversation_history

            # Если есть ошибки, готовим фидбек для следующей итерации
            last_validation_feedback = validation_feedback
            print(f"⚠️ Найдены проблемы: {last_validation_feedback[:200]}")

        print(f"❌ Не удалось получить валидный код после {self.max_iterations} итераций.")
        return current_code, False, self.conversation_history


if __name__ == "__main__":
    # Проверяем наличие модели
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        print("✅ Ollama найден")
    except Exception:
        print("❌ Ollama не установлен. Установите: brew install ollama")
        sys.exit(1)

    # Создаем агента
    agent = LuaCodeAgent(model_name="codellama:7b-instruct")

    # Тестовый запрос
    query = "Из полученного списка email получи последний"

    # Запускаем агента
    final_code, success, history = agent.run(query)

    print("\n" + "=" * 50)
    print("ФИНАЛЬНЫЙ КОД:")
    print("=" * 50)
    print(final_code)
    print("=" * 50)

    if success:
        print("✅ СТАТУС: Успешно сгенерирован и проверен")
    else:
        print("⚠️ СТАТУС: Сгенерирован с ошибками")

    # Сохраняем код в файл
    with open('output.lua', 'w') as f:
        f.write(final_code)
    print("💾 Код сохранен в output.lua")