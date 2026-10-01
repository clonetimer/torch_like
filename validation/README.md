# 0.9.0 验收证据阅读顺序

`release_validation.json` 是最终冻结源码的唯一测试总账：574/574，无失败、无跳过，31 个分组。
`release_*.log`、`release_*.xml` 是这轮全量结果；`release_ddp_*_rank_results.json` 来自实际双进程 worker。
`staged_smoke.*` 是清洁发布目录的额外 40 项复验，不叠加到 574。
`installed_*` 为源码外安装态 smoke、真实 DDP、示例训练和恢复。
`release_gates.*` 为 API/catalog/lazy/compileall 门禁与实际环境。
`package_integrity.json` 说明冻结源码和清洁 staging 的摘要一致，wheel 的276个包内文件逐一一致。
`change_manifest.json` 是相对0.8的源码改动台账。

历史文件 `development_registry_failure.log` 刻意保留开发阶段写死0.8.0版本断言的失败；它不属于最终574项结果。
`ruff_install_attempt.log` 记录工具安装失败，本次没有 Ruff clean/覆盖率门禁通过声明。
日志中出现的 /mnt/data、/tmp 路径是执行环境取证，不是用户本地必需目录。
CUDA/NCCL、多机、Windows和其他依赖版本未在本轮实测。
