# Relatório Técnico: Pregnancy Health Helper

## 1. Descrição do Fluxo Multimodal
A aplicação utiliza um fluxo de processamento **multimodal** que integra dados biométricos estruturados com dados não-estruturados provenientes de áudio. O fluxo é composto pelas seguintes etapas:
1.  **Ingestão:** Recebimento de biomarcadores (Idade, Pressão arterial, Glicose, Frequência Cardíaca e Temperatura) e áudio de entrevista em português.
2.  **Processamento Fisiológico:** Conversão de unidades (Celsius para Fahrenheit) e classificação de risco gestacional via modelo de Machine Learning.
3.  **Pipeline de Áudio:** 
    *   **Limpeza:** Redução de ruído espectral.
    *   **Acústica:** Extração de biometria vocal (Pitch, Volume, Taxa de Fala).
    *   **Linguística:** Transcrição (Local ou Cloud) seguida de análise de sentimento e classificação semântica.
4.  **Fusão:** Cruzamento dos scores de confiança da IA linguística com os indicadores físicos da voz para gerar diagnósticos de condições específicas.
5.  **Entrega:** Consolidação dos resultados em um objeto JSON retornado ao frontend para exibição e armazenamento local.

## 2. Modelos Aplicados em Cada Tipo de Dado

### Dados Fisiológicos (Biomarcadores)
*   **Modelo:** `RandomForestClassifier` (Scikit-Learn).
*   **Aplicação:** Classificação do nível de risco na gestação (Low, Mid, High) baseado em indicadores vitais.
*   **Treinamento:** Como a correlação entre os dados e o risco gestacional era evidente, o treinamento foi simples. Apenas uma normalização na coluna de risco foi necessária e com 100 estimators o accuracy_score ficou em 0.81

### Dados de Áudio (Processamento em Camadas)
1.  **Redução de Ruído:** `noisereduce` (Algoritmo de redução espectral).
    
    **Objetivo:** Remover ruídos de fundo e ruídos brancos para aumentar a precisão da transcrição.
1.  **Transcrição (Linguística):**
    *  **Transcrição Local:** `faster-whisper` (Modelo `large-v3-turbo` com quantização **INT8**).
    *  **Transcrição Cloud:** `gemini-3-flash-preview` (Google GenAI SDK). Utilizado para transcrição de alta fidelidade via API.
2.  **Análise de Sentimento:**
    *   **Modelo:** `nlptown/bert-base-multilingual-uncased-sentiment`. (Hugging Face).
    *   **Objetivo:** Classificar a polaridade da fala em uma escala de 1 a 5 estrelas, mapeadas para labels clínicos (ex: "Estresse Elevado").
3.  **Detecção de Condições (Inferência Zero-Shot):**
    *   **Modelo:** `facebook/bart-large-mnli` (Facebook).
    *   **Objetivo:** Utiliza **Zero-Shot Classification** para detectar sintomas sem a necessidade de treinamento específico para cada termo, analisando o contexto semântico completo da entrevista.
4.  **Extração de Features:** Biblioteca `librosa` para análise de Pitch (F0), RMS (Volume) e ZCR (Rouquidão).
   
## 3. O que a analise de áudio buscava encontrar, cruzando os dados:
A aplicação é capaz de correlacionar padrões de texto com características físicas da voz para identificar anomalias:

| Condição Detectada | Indicador Linguístico (NLP) | Indicador Acústico (Librosa) |
| :--- | :--- | :--- |
| **Depressão Pós-Parto** | Relatos de culpa, tristeza ou desânimo. | Afeto plano (baixa variabilidade de pitch) e pausas prolongadas. |
| **Fadiga Hormonal** | Termos como "exausta", "sono" ou "névoa mental". | Baixa energia vocal (amplitude reduzida) e tom de voz reduzido. |
| **Ansiedade Perinatal** | Preocupação constante, medo ou pânico. | Fala acelerada (Speech Rate > 0.8) e alta variabilidade de pitch. |
| **Refluxo (GERD)** | Contexto geral de desconforto gástrico. | **Índice de Rouquidão:** Alta taxa de cruzamento por zero (ZCR), indicando voz "arranhada". |
| **Violência Doméstica** | Medo, hesitação em contar ao parceiro, relatos de machucados. | Volume extremamente baixo (sussurro) e alta taxa de silêncio/hesitação. |

## 4. Resultados Obtidos e Exemplos de Anomalias Detectadas
O sistema obteve sucesso em identificar condições de saúde mental e física através da **Lógica de Fusão** (Linguística + Acústica).

### Exemplos de Anomalias Detectadas:

*   **Anomalia A: Depressão Pós-Parto**
    *   **Cenário:** Paciente relata sentimentos de culpa e tristeza.
    *   **Detecção:** O modelo BART detecta a intenção semântica (Score > 0.5) e a análise de Librosa identifica uma variabilidade de pitch extremamente baixa (< 50Hz) e fala lenta, caracterizando o "afeto plano".
    *   **Resultado:** Marcado como "Possíveis sinais de Depressão Pós-Parto".

*   **Anomalia B: Refluxo Gestacional (GERD)**
    *   **Cenário:** Paciente apresenta voz rouca durante o relato.
    *   **Detecção:** Independentemente do conteúdo do texto, o algoritmo acústico identifica um "**índice de Rouquidão**" (ZCR) superior a 0.12.
    *   **Resultado:** Marcado como "Sinais de Refluxo (GERD Gestacional)".

*   **Anomalia C: Ansiedade Perinatal**
    *   **Cenário:** Paciente fala de forma acelerada sobre medos constantes.
    *   **Detecção:** O modelo de sentimento classifica a fala como "Muito Negativa / Estresse Elevado" e a métrica de **Speech Rate** ultrapassa 80% de atividade vocal sem pausas.
    *   **Resultado:** Marcado como "Sinais de Ansiedade Perinatal".

*   **Anomalia D: Alerta de Violência Doméstica**
    *   **Cenário:** Paciente fala em volume muito baixo, hesitando em citar o parceiro.
    *   **Detecção:** Identificação de volume vocal crítico (< 0.01 RMS) combinada com classificação zero-shot para "relato de perigo ou machucados".
    *   **Resultado:** Emite alerta de "Indicadores de Alerta (Violência Doméstica)".
*   

## 5. Considerações
Como não é possível encontrar áudios de consultas pré natais reais pela internet, os testes tiveram de ser efetuados por meio das simulações de entrevistas pré natais disponíveis no Youtube. Para que o método tenha validade, é indispensável que sejam feitos testes com entrevistas reais. De preferência isolando apenas o áudio da paciente (retirando as falas do entrevistador) para que a variação de pitch, volume e rouquidão da fala possam ser medidas somente da pessoa que está sendo avaliada.
