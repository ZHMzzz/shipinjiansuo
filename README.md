# 寻帧 · 本机视频检索

基于 WISE 2 的视频检索研究项目，支持用中文、参考图片和已登记人物查找视频，并跳转到命中时间。自然语言与图文检索使用 Qwen3-VL-Embedding 2B，物体图片使用 OWLv2，人脸使用 InsightFace。

## 主要功能

- **检索视频**：描述场景或动作、上传物体参考图，也可组合图片与文字查询。
- **查找人物**：按已登记姓名检索，可叠加场景或图片条件；每人支持多张登记照。
- **管理视频库**：按文件名筛选，批量上传、查看处理进度、失败重试及删除视频与索引。
- **定位播放**：从命中时间播放，优先使用原视频，无法解码时回退到压缩版。

## 快速开始

当前仅适配 **Apple Silicon Mac**，已在 **M5 / 32 GB** 验证。Windows、Linux、Intel Mac 尚未适配。

先准备 Xcode Command Line Tools、Homebrew、Python 3.11、Node 22 和 libmagic；具体命令见 [首次安装与运行](docs/首次安装与运行.md)。随后执行：

```sh
git clone --depth 1 https://github.com/ZHMzzz/shipinjiansuo.git
cd shipinjiansuo
./scripts/setup.sh --check
./scripts/setup.sh
./scripts/index.sh
./scripts/start.sh
```

打开 **[http://127.0.0.1:9670/mvp/](http://127.0.0.1:9670/mvp/)**。首次安装会下载模型、构建前端并建立演示视频索引；首次启动后还会补建 Qwen 片段，等待页面中的索引处理提示消失后再做自然语言查询。进度及错误见 `logs/server.log`。

日常只需启动或停止服务：

```sh
./scripts/start.sh
./scripts/stop.sh
```

## 文档

- [首次安装与运行](docs/首次安装与运行.md)：环境准备、操作步骤、验收和排错。
- [安装验证记录](docs/首次安装验证.md)：实测结果及验证范围。
- [功能与维护](docs/功能与维护.md)：检索规则、性能开关、模型配置及源码入口。
- [视频上传与视频库](docs/视频上传与视频库.md)：上传处理、数据位置和版本管理。
- [实验说明](experiments/README.md)：历史报告、源码及实验数据恢复。

## 数据与使用范围

仓库附带 14 个初始视频和参考图，来源见 [素材清单](data/manifest.json)。模型、索引、上传视频、人物登记及实验产物不随源码交付。推荐浅克隆或下载最新 ZIP；完整 Git 历史仍包含旧实验文件。

这是小规模研究演示，片段边界不等于精确动作边界，相似度也不代表正确概率；尚未验证大规模、跨摄像头或长期稳定运行。服务仅供本机访问。

WISE 代码采用 Apache-2.0；InsightFace 预训练模型当前按非商业研究使用。素材权利归各自作者。
