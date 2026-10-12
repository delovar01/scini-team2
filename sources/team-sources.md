# Источники

**`references.bib`** — 100 работ по косвенным промпт-инъекциям и защитам агентов, собраны 16.09, каждая сверена с первоисточником (название, авторы, год, площадка). Ключи из него ставим в текст: `[greshake2023not]`. Разделы списка: основы агентов и RAG (1–6), прямые инъекции и джейлбрейки (7–16), косвенные инъекции (17–26), каналы доставки и RAG (27–37), защиты промптом (38–44), обучением (45–52), архитектурные (53–63), детекторы (64–73), бенчмарки и методология (74–92), русскоязычные (93–100).

Правило курса: источники из Google Scholar, eLibrary, Scopus, Web of Science; arXiv можно, но проверяй, вышла ли статья после рецензии.

Ниже — то, что команда прислала в чат 7–8.10 и чего нет в `references.bib`. Савелий переносит в bib с полными данными до 20.10.

## Павел — обзор атак (eLibrary)

1. Промпт-инъекция: способы реализации и методы защиты // Безопасность. Управление. Искусственный интеллект. 2026. Т. 5, № S5. С. 117–127. https://elibrary.ru/item.asp?id=91844196
2. Анализ атак на большие языковые модели // Научное сообщество студентов XXI столетия. Технические науки: материалы CXLV студ. междунар. науч.-практ. конф. Новосибирск, 2025. С. 32–43. https://elibrary.ru/item.asp?id=80293290
3. Уязвимости больших языковых моделей: анализ и методы защиты // Проблемы информатики. 2025. № 3 (68). С. 17–33. https://elibrary.ru/item.asp?id=83162635
4. HAMSA: автоматизированный метод атак компактных выровненных моделей // Доклады РАН. Математика, информатика, процессы управления. 2025. Т. 527, № S. С. 449–458. https://elibrary.ru/item.asp?id=83189225
5. Модель угроз безопасности LLM-приложений: prompt injection, утечки данных и атаки на RAG-контекст // Всероссийская студенческая конференция: ст. III Всерос. науч.-практ. конф. Пенза, 2026. С. 49–51. https://elibrary.ru/item.asp?id=91631015
6. Prompt Injection Attacks in Large Language Models and AI Agent Systems — дописать выходные данные (в чате обрезано)

Авторов у 1–5 в чате нет — добавить с eLibrary.

## Алексей — атаки на RAG

1. Zou W. et al. PoisonedRAG — уже есть в bib как [zou2024poisonedrag]
2. Practical Poisoning Attacks against Retrieval-Augmented Generation (CorruptRAG). https://doi.org/10.1145/3750555.3811900
3. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models. https://arxiv.org/abs/2406.00083
4. HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models. https://arxiv.org/abs/2410.22832
5. Human-Imperceptible Retrieval Poisoning Attacks in LLM-Powered Applications. https://doi.org/10.1145/3663529.3663786
6. Tricking Retrievers with Influential Tokens: An Efficient Black-box Corpus Poisoning Attack // NAACL 2025. https://aclanthology.org/2025.naacl-long.210/
7. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents // USENIX Security 2025. https://www.usenix.org/system/files/usenixsecurity25-shafran.pdf
8. The Hidden Threat in Plain Text: Attacking RAG Data Loaders (PhantomText Toolkit). https://doi.org/10.1145/3733799.3762976
9. Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection — дописать выходные данные

## Азат — защиты и уязвимый код

1. Pearce H. et al. Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions. arXiv:2108.09293
2. Perry N. et al. Do Users Write More Insecure Code with AI Assistants? arXiv:2211.03622
3. Sandoval G. et al. Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants. arXiv:2208.09727
4. Spracklen J. et al. We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs. arXiv:2406.10279
5. Tihanyi N. et al. How Secure is AI-generated Code: A Large-Scale Comparison of Large Language Models. arXiv:2404.18353

Плюс по защитам от косвенных инъекций — брать из bib, разделы 38–73.

## Савелий — систематизация и модель угроз

1. Zhang J., Bu H., Wen H. et al. When LLMs meet cybersecurity: a systematic literature review // Cybersecurity. 2025. Vol. 8. Art. 55. https://link.springer.com/article/10.1186/s42400-025-00361-w
2. Cui T., Wang Y., Fu C. et al. Risk Taxonomy, Mitigation, and Assessment Benchmarks of Large Language Model Systems. 2024. https://arxiv.org/abs/2401.05778
3. Tete S. B. Threat Modelling and Risk Analysis for Large Language Model (LLM)-Powered Applications. 2024. https://arxiv.org/abs/2406.11007
4. Li M. … — в чате обрезано, дописать

## Степан — введение и эксперимент

Всё из bib: [greshake2023not], [zhan2024injecagent], [debenedetti2024agentdojo], [yi2023benchmarking], [zhan2025adaptive], [hines2024defending], [shi2025progent], [zheng2023judging], [owasp2025llm], [mudarova2024], [evgrafov2026].
