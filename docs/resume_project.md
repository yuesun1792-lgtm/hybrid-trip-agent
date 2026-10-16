# EuroTrip Agent：简历项目经历素材

更新日期：2026-10-16。待正式简历加入仓库后再合并；当前尚未修改简历原文件。
以下描述以已实现代码为依据，不包含未经测量的效果或性能提升。

## 中文：AI 应用 / 后端方向

**EuroTrip Agent｜支持 RAG 与 MCP 工具调用的旅行规划系统**

Python · FastAPI · Pydantic · NumPy · OpenAI / Anthropic · MCP · Streamlit

- 设计统一受控规划流程，支持多轮需求澄清、目的地比较、结构化行程生成与局部修改；复用候选概要分析和选中城市详细分析能力。
- 基于覆盖 22 个欧洲目的地的 Wikivoyage 语料构建本地检索，关联活动与稳定证据 ID，并通过独立展示层处理英文检索与多语言回复。
- 实现有边界的模型原生工具调用节点，通过本地 MCP 客户端/服务端接入 Geoapify 地点搜索与详情；校验候选来源、城市与国家，避免模型生成坐标。
- 使用 Day/Activity 稳定 ID、修改作用域与版本快照保护未修改活动，结合参数/结果校验、调用预算、JSON/JSONL 轨迹和多轮场景评测验证行为。

篇幅有限时保留第一、第三、第四条；第二条适合偏 RAG 的岗位。

## 中文：数据平台 / 数据工程方向

**EuroTrip Agent｜旅行知识检索与可追踪规划服务**

Python · FastAPI · NumPy · Pydantic · RAG · MCP

- 构建覆盖 22 个欧洲目的地的旅行知识处理与检索链路，将离线语料索引接入在线规划服务，以稳定证据 ID 关联检索结果与下游活动。
- 定义约束、活动、地点、证据和修改请求等类型化数据契约，在提交前校验天数、城市、引用关系和保留约束，通过版本检查拒绝过期修改。
- 接入真实地点服务并处理搜索/详情 ID 规范化差异、代理连接和错误分类；记录模型用量、工具输入输出与异常，支持基于轨迹定位上下游问题。
- 建立离线回归和多轮场景评测，分离真实服务验证与模拟依赖测试，输出计划快照、断言结果、调用次数和费用估算，支持自动化执行。

## English: AI application / backend

**EuroTrip Agent — RAG-grounded travel planning with bounded MCP tool use**

Python · FastAPI · Pydantic · NumPy · OpenAI / Anthropic · MCP · Streamlit

- Designed a shared planning workflow for multi-turn clarification, destination comparison, structured itinerary generation, and scoped revisions, with overview analysis for candidates and detailed research for the selected city.
- Integrated a local Wikivoyage retrieval corpus covering 22 European destinations, linking activities to stable evidence IDs and separating canonical planning state from multilingual presentation.
- Implemented bounded native model tool calling through a local MCP client/server backed by Geoapify search and details APIs; validated provider records, city/country scope, and lookup provenance before attaching coordinates.
- Protected unchanged activities using stable IDs, scoped merges, version snapshots, and optimistic concurrency checks; added execution budgets, JSON/JSONL traces, offline regression tests, and automated multi-turn scenario evaluation.

## 面试边界

- 主链路是受控工作流；模型动态决策集中在有限节点，并非任意工具自主循环。
- MCP 已用于真实地点解析；RAG 仍在本地工具注册表中，交通仍使用模拟估算。
- 尚未核实实时营业时间、交通班次、门票价格，未接入预订能力。
- 会话与版本仓库仍是单进程内存；轨迹落盘不等于持久执行或 checkpoint。
- 未接入 LangGraph、多 Agent 或 TREK 协作/记账，不能写成已实现。
- 未声称准确率、延迟、吞吐量或成本相对基线提升；测试数量属于开发快照，不是业务效果。

## 30 秒介绍

> 我做了一个支持多轮对话的旅行规划系统，把目的地比较、行程生成和局部修改统一到受控流程中，用 RAG 提供旅行知识，并让地点节点通过 MCP 自主查询真实地点服务。重点是让模型的建议可以被程序校验：活动有稳定 ID，局部修改保留其他活动，坐标来自工具记录，每次运行都有轨迹和场景断言。当前已验证真实工具调用，后续重点是持久化和交通可行性校验。
