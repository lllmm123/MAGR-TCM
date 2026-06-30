Data Confidentiality Notice

The full Medical Knowledge Graph (comprising over 10,000 entities and 30,000 relationships) utilized in this project is currently withheld. This is in strict compliance with medical data privacy policies and the ongoing peer-review process of our associated academic manuscript.

Please note that no dataset files are included in this repository. However, the Multi-Agent RAG architecture and the custom Dynamic Beam Search algorithms are highly decoupled and designed to be fully compatible with standardized JSON/Neo4j graph structures. Researchers and developers can independently initialize the reasoning engine by plugging in their own domain-specific knowledge graphs.

# 🌿 TCM-GYN-MDT — 中医妇科多智能体智能决策系统

基于 **多智能体协作(Multi-Agent)** 与 **知识图谱(GraphRAG)** 的中医妇科家庭辅助分诊平台。

## 系统架构

```
用户输入 → Agent 1(症状提取) → Agent 2(GraphRAG检索+初稿生成)
                                  ↓
                             Agent 3(知识图谱锚定)
                                  ↓
用户 ← Agent 4(双引擎辩论质控+输出)
```

- **Agent 1 — 临床语义标定**: 基于 Moonshot Kimi 大模型，从日常语言中提取 11 类中医临床实体
- **Agent 2 — 弹性束搜索**: 知识图谱多跳路径检索 + DeepSeek 生成循证初稿
- **Agent 3 — 图谱推理引擎**: 基于 Neo4j 结构的中医妇科知识网络，支持 BFS 邻域展开与多跳隐性路径寻踪
- **Agent 4 — 异构成文质控**: 双引擎动态评分辩论 (RAGAS → Faithfulness/Relevancy/Recall/Precision)，阻断事实幻觉

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Streamlit + PyVis 知识图谱可视化 |
| 后端 | Python 多智能体架构 |
| NLP | spaCy (zh_core_web_sm) + jieba 分词 |
| 知识图谱 | 自建中医妇科知识图谱 (~6MB JSON, Neo4j 结构) |
| 大模型 | DeepSeek API + Moonshot Kimi API |
| 语义评估 | RAGAS 框架 (Faithfulness/Relevancy/Recall/Precision) |
| 模糊匹配 | RapidFuzz + Levenshtein |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 spaCy 中文模型
python -m spacy download zh_core_web_sm

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 4. 启动
streamlit run app.py
```

## 项目亮点

- 🧠 **多智能体协作架构**: 四个 Agent 分工明确，模拟临床会诊流程
- 🕸️ **知识图谱驱动**: 自建中医妇科领域知识网络，支持多跳路径推理与可视化溯源
- 🛡️ **反幻觉机制**: 双引擎交叉辩论 + RAGAS 动态评分质控
- 📜 **白盒化推理**: 每一次诊断结果都可追溯至图谱推理路径

## 免责声明

本项目仅供学习研究使用，不构成医疗建议。如有身体不适，请及时前往正规医院就诊。
