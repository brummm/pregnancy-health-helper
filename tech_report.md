# Relatório Técnico: Pregnancy Health Helper (Auxílio Gestacional Multimodal)

## 1. Descrição do Fluxo Multimodal
A aplicação opera através de um fluxo de dados **multimodal**, integrando indicadores fisiológicos quantitativos (biomarcadores) com dados qualitativos extraídos de áudio (entrevistas clínicas). 

O fluxo segue as seguintes etapas:
1.  **Ingestão de Dados (Frontend):** O usuário insere dados biomarcadores (Idade, Pressão Arterial, Glicose, Frequência Cardíaca e Temperatura) e faz o upload de um arquivo de áudio da entrevista em português.
2.  **Orquestração (Backend):** O backend em **Flask** recebe a requisição multipart e divide o processamento em dois pipelines paralelos: **Pipeline de Risco Fisiológico** e **Pipeline de Análise de Áudio**.
3.  **Fusão e Diagnóstico:** Os resultados de ambos os pipelines são consolidados. A análise linguística do áudio é cruzada com métricas acústicas para detectar anomalias específicas (Depressão, Ansiedade, etc.), enquanto os biomarcadores definem o nível de risco gestacional geral.
4.  **Persistência e Exibição:** Os resultados são retornados em JSON e armazenados no `localStorage` do navegador para consulta histórica.

---

## 2. Modelos Aplicados por Tipo de Dado

### A. Dados Fisiológicos (Biomarcadores)
Para a classificação do nível de risco da gravidez (Baixo, Médio ou Alto Risco), foi treinado um modelo dedicado:
*   **Modelo:** `RandomForestClassifier` (Scikit-Learn).
*   **Dataset:** Treinado sobre o *Maternal Health Risk Data Set* (via Kaggle).
*   **Processamento:** O backend realiza a conversão de temperatura (Celsius para Fahrenheit) e normalização dos dados antes da inferência.

### B. Dados de Áudio (Processamento Multiestágio)
O pipeline de áudio é o componente mais complexo da aplicação, utilizando modelos de última geração:

1.  **Limpeza e Limpeza de Ruído:**
    *   **Modelo:** `noisereduce` (Algoritmo de redução espectral).
    *   **Objetivo:** Remover ruídos de fundo e ruídos brancos para aumentar a precisão da transcrição.

2.  **Transcrição (Linguística):**
    *   **Opção Local:** `faster-whisper` (Modelo `large-v3-turbo` com quantização **INT8**). Oferece alta precisão em português brasileiro com eficiência em CPU.
    *   **Opção Cloud:** `Gemini 1.5 Flash / 3-Flash-Preview` (Google Generative AI). Utilizado para transcrição de alta fidelidade via API.

3.  **Análise de Sentimento:**
    *   **Modelo:** `BERT-base-multilingual-uncased-sentiment` (Hugging Face).
    *   **Objetivo:** Classificar a polaridade da fala em uma escala de 1 a 5 estrelas, mapeadas para labels clínicos (ex: "Estresse Elevado").

4.  **Detecção de Condições (Inferência Zero-Shot):**
    *   **Modelo:** `BART-large-mnli` (Facebook).
    *   **Objetivo:** Utiliza **Zero-Shot Classification** para detectar sintomas sem a necessidade de treinamento específico para cada termo, analisando o contexto semântico completo da entrevista.

5.  **Extração de Features Acústicas:**
    *   **Biblioteca:** `librosa`.
    *   **Métricas:** F0 (Pitch), Variabilidade de Tom, Intensidade (RMS), Zero Crossing Rate (ZCR) e Speech Rate (Taxa de Fala).

---

## 3. Resultados Obtidos e Anomalias Detectadas
A aplicação é capaz de correlacionar padrões de texto com características físicas da voz para identificar anomalias:

| Condição Detectada | Indicador Linguístico (NLP) | Indicador Acústico (Librosa) |
| :--- | :--- | :--- |
| **Depressão Pós-Parto** | Relatos de culpa, tristeza ou desânimo. | Afeto plano (baixa variabilidade de pitch) e pausas prolongadas. |
| **Fadiga Hormonal** | Termos como "exausta", "sono" ou "névoa mental". | Baixa energia vocal (amplitude reduzida) e tom de voz reduzido. |
| **Ansiedade Perinatal** | Preocupação constante, medo ou pânico. | Fala acelerada (Speech Rate > 0.8) e alta variabilidade de pitch. |
| **Refluxo (GERD)** | Contexto geral de desconforto gástrico. | **Índice de Rouquidão:** Alta taxa de cruzamento por zero (ZCR), indicando voz "arranhada". |
| **Violência Doméstica** | Medo, hesitação em contar ao parceiro, relatos de machucados. | Volume extremamente baixo (sussurro) e alta taxa de silêncio/hesitação. |

### Exemplo de Anomalia Detectada:
*   **Entrada:** Áudio onde a paciente diz *"Eu me sinto muito sobrecarregada, cansada do bebe e triste, sem energia nenhuma"*.
*   **Processamento:** O modelo BART identifica alta confiança (score > 0.5) para o rótulo de depressão. O pipeline acústico confirma uma variabilidade de pitch abaixo de 50Hz.
*   **Resultado:** O sistema emite um alerta de **"Possíveis sinais de Depressão Pós-Parto"**, mesmo que os biomarcadores físicos (pressão, glicose) estejam normais.

---

## 4. Stack Tecnológica Resumida
*   **Linguagens:** Python 3.10+, TypeScript.
*   **Frameworks:** Flask (Backend), React 19 (Frontend).
*   **IA/ML:** PyTorch, Transformers, Faster-Whisper, Scikit-Learn.
*   **Infraestrutura:** Docker & Docker Compose (Isolamento de dependências e FFmpeg).
*   **APIs Externas:** Google Gemini API (Opcional).
