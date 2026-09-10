# 第三方内容与许可

本项目的非商业源码许可只覆盖有权授权的原创部分，不替换第三方许可证，也不授予外部服务账号或数据权限。

## 随源码分发的素材

- **Geist 字体**：`travel-agent/apps/web/public/fonts/Geist-*.woff2`，版权归 The Geist Project Authors（2024），采用 SIL Open Font License 1.1。完整许可保留于 [OFL.txt](travel-agent/apps/web/public/fonts/OFL.txt)，[上游项目](https://github.com/vercel/geist-font)。
- **项目标志与界面图标**：维护者已确认拥有公开分发权限。它们不表示任何第三方服务商为本项目背书。
- **首页视频**：仓库不分发视频副本，也不内置个人素材账户地址。`VITE_HOME_BACKGROUND_URL` 由部署者配置，部署者须取得相应使用权限。

## 软件依赖

仓库不包含 `node_modules`、Python 虚拟环境或生产构建。版本与依赖范围见 `travel-agent/pyproject.toml`、`travel-agent/requirements-agent.lock`、`travel-agent/apps/web/package.json` 和 `pnpm-lock.yaml`。

React、React Router、Vite、Markdown 处理库、FastAPI、Pydantic、SQLAlchemy、LangGraph、cryptography、Redis 客户端、阿里云 SDK 及其他直接或传递依赖，分别适用其发行包所附许可证。此处不是将这些依赖统一重新许可，也不是全部依赖版权声明的替代品。

**GSAP 使用自己的 [Standard License](https://gsap.com/community/standard-license/)，不是 MIT。** 请按该许可使用和分发，不要将其标注为本项目原创或移除版权声明。

制作或再分发部署包时，应同时保留所包含第三方代码、字体和图标要求的许可证与版权声明，并根据实际打包内容核查，不应只复制本项目的 LICENSE。

## 实时服务和数据

模型、高德地图、FlyAI、天气、短信等外部服务的账号、额度、商标和数据使用权不随本仓库授予。部署者需要自行确认相应服务条款、授权范围与费用。

运行中查询得到的地点图片、地图底图、商家信息和报价，不作为本项目原创素材再授权。未经授权或脱敏，不应将原始服务响应和用户记录加入公开仓库。
