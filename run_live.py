import os
import json
from dataclasses import asdict

# Принудительно включаем боевой режим до импорта конфигурации
os.environ["USE_STUB_RAG"] = "false"
os.environ["USE_STUB_RERANKER"] = "false"
os.environ["USE_STUB_LLM"] = "false"
os.environ["LLM_MODE"] = "real"

# Теперь импортируем оркестратор (он подтянет боевой конфиг)
from src.generation.orchestrator import run_agent

def main():
    print("=== Интерактивный тест RAG-ассистента (End-to-End) ===")
    print("Работают реальные модели поиска и генерации.")
    print("Введите 'exit', 'quit' или 'выход' для завершения.\n")

    while True:
        try:
            query = input("\nВВЕДИТЕ ЗАПРОС: ").strip()
            if query.lower() in ("exit", "quit", "выход", "q"):
                print("Завершение работы.")
                break
            if not query:
                continue

            print("\n🔄 Обработка запроса...")
            
            # Запускаем агента (можно передать event_date="YYYY-MM-DD", если нужно)
            result = run_agent(query)
            
            print("\n" + "="*60)
            print(f"🎯 ИНТЕНТ: {result.intent}")
            print(f"🚫 ОТКАЗ (Refused): {result.refused}")
            print("\n💡 ОТВЕТ:")
            print(result.final_answer)
            
            print("\n📚 ИСТОЧНИКИ:")
            if result.sources:
                for source_id, meta in result.sources.items():
                    print(f"  [{source_id}] {meta['document_title']}")
                    print(f"        Путь: {' -> '.join(meta['structure_path'])}")
                    print(f"        Score: {meta['score']:.3f} | Chunk: {meta['chunk_id']}")
            else:
                print("  Нет релевантных источников.")
            
            print("\n🔍 ПОЛНЫЙ ТРЕЙС (Trace):")
            # Сериализуем трейс в читаемый JSON
            trace_dict = [asdict(step) for step in result.trace]
            print(json.dumps(trace_dict, indent=2, ensure_ascii=False))
            print("="*60)

        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            break
        except Exception as e:
            print(f"\n❌ Ошибка выполнения пайплайна: {e}")

if __name__ == "__main__":
    main()