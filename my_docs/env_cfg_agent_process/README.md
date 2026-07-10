# AI Agent 生成 env_cfg 的流程设计与学习资料

这个目录是知识库入口，推荐直接打开 `index.html`。

- `index.html`：总览与推荐阅读顺序
- `architecture.html`：skill / workflow / orchestrator / AI API 的架构分层
- `architecture_evolution.html`：从单文件 orchestrator 到 template generator 和模块化拆分的开发演进
- `schema_spec.html`：Schema 与 Spec 的关系
- `workflow.html`：Spec Builder、Generator、Validation Gate
- `orchestrator.html`：Orchestrator 的作用、命令和伪代码
- `orchestrator_code.html`：`orchestrator.py` 的函数职责和代码流程导读
- `module_flow.html`：每个 Python 模块职责，以及 orchestrator 何时调 AI、何时走 presets/template 的流程图
- `adaptation_test_gate.html`：缺 exact template 时的 AI adaptation candidate/generator 流程，以及自动测试 gate 如何替代人工核查
- `preset_enrichment.html`：为什么需要 robot/object preset enrichment，以及它如何补齐 spec
- `template_generator.html`：env_cfg 薄封装、task 注册和启动命令如何由代码确定性生成
- `placement_workflow.html`：object.init_state 和 target_pose 如何按机器人工作空间自动计算
- `skill_files.html`：`env_cfg_skills` 执行材料说明
- `presets_prompts.html`：presets/ 字段清单与 prompts/ 职责边界参考
- `dev_log.html`：开发日志，记录工程化评审问题、修复顺序和对应代码变更
- `roadmap.html`：从当前雏形到产品化的路线
- `schemas/`：知识库侧的 schema 备份/学习版本
