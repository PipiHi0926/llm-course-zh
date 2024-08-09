<div align="center">
  <h1>🤖 大型語言模型中文分享整理 - 2 💻</h1>
  <p align="center">
    ✍️ <a href="https://hackmd.io/@pputzh5cRhi6gZI0csfiyA/H1ejIyxHR"> 作者: 鄭永誠</a> • 
    ✉️ <a href="mailto:jason0304050607@gmail.com">信箱</a> 
    🧑‍🤝‍🧑 <a href="https://www.dalabx.com.tw//"> 合作夥伴1: 紫式大數據決策 </a>
    👫 <a href="https://moraleai.com/"> 我的朋朋2: Morale AI </a>
  </p>
</div>
<br/>

內容簡介:
1. 🍻 **LLM基礎改念:** 我會整理一些LLM的需求知識，但由於這裡是實作課資源不會講述太多，有興趣請聯繫我
2. 🛠️ **LLM相關工具:** 以下內容全基於python實踐，同時會分享相關資源、套件、開源API...
3. 💬 **LLM系統架構:** 會帶你由淺入深，慢慢了解部屬LLM系統(多Agent)的方向和好用工具


這個分享內容宗旨:
1. 🧩 **讓你好上手:** 提供最簡單的、盡可能可複製即用的code，讓新手也能盡可能快速入門 (而且是中文XD)
2. 🎈 **讓你免費玩:** 全基於開源資源，讓你能夠無痛體驗LLM的功能和操作
3. 😊 **讓你喜歡上:** 盡量提供簡單有趣的小例子，讓你也能喜歡LLM可帶來的運用

範例使用版本/輔助工具:
- Python 3.12.4
- 語言模型主要使用 llama-3.1-70b-versatile
- 個人主要使用 IDE: VScode
- 搭配工具 寫程式大幫手 [Copilot](https://github.com/features/copilot)
- 其他: 使用 [pylint](https://code.visualstudio.com/docs/python/linting) 擴充套件來管理 Python 程式碼的風格

主要資料來源:
  💻 <a href="https://github.com/">資料來源1: 偉大的Github</a> • 
  🤗 <a href="https://huggingface.co/">資料來源2: 偉大的抱抱臉</a> • 

---------------
  

## 課程內容
![alt text](images/image.png)
(因分享會有搭配我的簡報才會以這個架構講述，實際上這些課程無完全連貫性)

| 主題 | 簡介 | 類別 | Notebook |
|----------|-------------|----------|----------|
| C0-前置作業與基礎工具|建立虛擬環境、基礎python輔助工具| 前置作業 |[C0](C0-Basic_info.ipynb)|
| C1-簡單使用範例|Groq操作、程式實踐基礎問答| 基礎課程 |[C1](C1-Get_start_with_groq.ipynb)|
| C2-立刻部屬簡易系統|Gradio快速實踐系統介面、即時對話系統| 基礎課程 |[C2](C2-Create_llm_ui.ipynb) |
| C3-已結合LLM的一些開源工具|智能網頁爬蟲Scrapegraph-ai| 額外分享 |[C3](C3-Ai_tools.ipynb)|
| C4-進階RAG操作|Reranker概念和效果| 上次課程補充 |[C4](C4-Advanced_rag.ipynb)|
| C5-實踐LLM服務Agent流程-1|基於Langchain架構下的LangGraph實踐| 進階課程 |[C5](C5-Agent_flow.ipynb)|
| C6-實踐LLM服務Agent流程-2|基於Langchain架構下的LangGraph實踐| 進階課程 ||
| C7-將Agent流程進行管控|使用langsmith來管理、更清楚瞭解建立的流程| 進階課程 ||
| C8-fine-tuned簡易操作範例|使用Unsloth簡易實踐fine tuned(只放程式碼、不實際運行)| 進階課程 ||


## 先備知識
### 1. LLM是什麼 ?
- 大型語言模型 (Large Language Model) 的簡稱
- 你可以把他理解成是一個模型，能根據輸入的文字生成文字回傳，就像在做文字接龍一樣
- 背後深度學習, Tokenization, embedding, attention機制, Transformer 等相關介紹詳見以前分享
![alt text](images/image-2.png)
![alt text](images/image-3.png)
![alt text](images/image-5.png)

### 2. Hugging Face 🤗 是什麼?
- 你可以把他理解成AI界的Github
- 使用者可以在上邊發表和[共享預訓練模型](https://huggingface.co/docs/transformers/model_sharing)、資料集和展示檔案
- 許多[模型排名](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)、[程式範例](https://huggingface.co/docs/transformers/llm_tutorial)都可在上面找到

### 3. Llama 🦙 是什麼 ?
- Llama是Meta開發的一系列大型語言模型，如llama2, llama3, llama3.1
- 這些語言模型都是免費的!!也能自行去做模型參數微調訓練(Fine-tuned)
- 至於其他常見付費LLM則包含[Open AI](https://openai.com/index/openai-api/), [Claude](https://www.anthropic.com/api)系列...

### 4. Ollama 是什麼 ?
- 開源的本地端LLM平台
- 允許用戶在自己的電腦上運行和調用多種開放原始碼的語言模型
- 若要部屬自己公司/組織內部的LLM，可以運用其資源
- 建議可搭配[Open WebUI](https://docs.openwebui.com/)、[AnythingLLM](https://anythingllm.com/)實踐UI操作介面和管理

### 5. LangChain 是什麼 ?
- [LangChain](https://python.langchain.com/v0.2/docs/introduction/)是LLM框架，目的在簡化使用大型語言模型（LLMs）開發應用程序的過程
- 你可以簡單理解成他是個工具箱，把LLM操作過程可能需要的工具、外部數據源整合起來
- 有這個工具箱，你就能更方便的調用他撰寫python llm相關程式
- 其他優勢包含很多延伸服務和工具也基於他被開發出來 (如後面會講到的[LangGraph](https://langchain-ai.github.io/langgraph/))
- 對於新手而言，他官網上也有非常大量[範例程式](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/)輔助你實踐llm
- 其他常用框架還有LlamaIndex，其更擅長處理文本(e.g.非結構資訊)、自定義知識庫、有多種索引查詢功能

### 6. Transformer 是什麼 ?
- Transformer是一種深度學習模型架構，在多種自然語言(NLP)任務處理表現出色
- 為當今(2024)生成式AI的浪潮重大突破的核心技術概念

![alt text](images/image-6.png)

- ChatGPT 裡面的 "T"，指的就是Transformer喔!

![alt text](images/image-7.png)


## 課程套件/工具/架構運用
- C0: 可略過
- C1: Langchain, Groq
- C2: Gradio
- C3: 可略過
- C4: Sentence-transformer, Reranker (Jina)
- C5: Langchain, LangGraph

