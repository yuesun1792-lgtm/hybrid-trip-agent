# Python 目录分类标准

按“主要职责 + 共同变化原因”分类：同一能力中经常一起修改的模块放在同一包；跨功能共享的契约放入 domain，运行支撑放入 runtime。文件名保留，便于追踪移动历史。此次只调整组织和引用，不重写算法或强制消除现有跨包依赖。

## 模块归属

| 目录 | 职责 | 文件 |
| --- | --- | --- |
| `backend/app/conversation/` | 对话入口、会话状态、意图路由及语言呈现。 | `chat.py`, `conversation_state.py`, `language.py`, `strategy.py` |
| `backend/app/domain/` | 跨功能共享的数据契约、字段规范化和目的地名称。 | `models.py`, `planning_models.py`, `place_models.py`, `destination_names.py`, `field_validation.py` |
| `backend/app/planning/` | 当前行程业务流程：任务构建、目的地分析、日程生成、修订、校验、版本存储、渲染和费用规则。 | `task_builder.py`, `unified_planner.py`, `destination_analysis.py`, `plan_builder.py`, `revision.py`, `plan_validation.py`, `plan_store.py`, `plan_renderer.py`, `planning_engine.py`, `costs.py` |
| `backend/app/rag/` | 知识检索、重排、检索研究和基于证据的候选发现。 | `knowledge.py`, `rag_enhancements.py`, `reranker.py`, `candidate_discovery.py` |
| `backend/app/tool_calling/` | 通用工具实现、工具 schema、分发和受控执行；不局限于 MCP。 | `tools.py`, `tool_executor.py`, `currency_tool.py` |
| `backend/app/places/` | 地点解析这一外部集成能力：Geoapify provider、MCP client/server 和模型驱动的地点解析节点。 | `places_mcp_client.py`, `places_mcp_server.py`, `place_provider.py`, `place_resolution.py` |
| `backend/app/llm/` | 模型供应商适配、调用结果和 token 费用统计，以及现有 agent prompts。 | `llm.py`, `llm_usage.py`, `prompts.py` |
| `backend/app/runtime/` | 跨业务的运行预算、超时和调用轨迹。 | `execution.py`, `trajectory.py` |
| `backend/app/legacy/` | 早期规划算法、旧 agent 和兼容入口。保留现有调用行为，当前流程仍会复用其中部分回退函数。 | `agent.py`, `planner.py`, `explainer.py`, `candidate_ranker.py`, `fixed_itinerary.py`, `rag_plan_builder.py` |

## 放置新代码的规则

- 行程生成、修订及业务校验放入 planning；检索与证据排序放入 rag。不要因为规划代码调用了检索就将整个规划器放入 rag。
- 通用工具注册、schema 和调用控制放入 tool_calling；地点 MCP 协议与服务商适配放入 places。以后新增其他 MCP 业务集成时，可建立独立的 integrations 子包，再按具体能力组织。
- 通用模型 API 接入放入 llm；业务专属的提示和编排继续靠近对应业务模块。
- 共享数据模型放入 domain；预算、轨迹等执行支撑放入 runtime。单个功能内部的辅助函数留在所属功能内，避免建立含义模糊的 utils。
- 新功能优先进入当前职责包；legacy 只维护已有兼容代码。legacy 并不代表代码没有调用者，删除前仍需检查依赖。
- frontend、scripts、tests 已分别承担界面、离线/运维命令、回归测试职责，因此保留现有目录。测试文件按原测试范围命名，本次不额外拆分。

## 导入与启动

- 包内外统一使用明确的绝对导入，例如 `from backend.app.rag.knowledge import search_travel_knowledge`。子包的 `__init__.py` 不主动加载子模块。
- HTTP 启动命令保持 `uvicorn backend.app.main:app --reload`。main.py 作为装配入口，config.py 保留在根目录，项目数据路径不变。
- MCP 服务的新命令为 `python -m backend.app.places.places_mcp_server`，客户端的子进程启动路径已同步更新。
- scripts 下的现有命令保持不变；测试中的 mock/patch 字符串也使用新路径。
- 旧的扁平 Python 导入路径已迁移，不额外保留同名转发文件；仓库外调用方需要同步更新导入。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -p no:capture -q tests
python -m scripts.test_places --discover
python -m scripts.eval_scenarios
```
