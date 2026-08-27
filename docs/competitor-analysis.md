# 竞品分析

调研 GitHub 上现有的 AI 生成视频质量评估类开源项目，明确 video-eval 的差异化定位。

## 一、GitHub 现有项目全景

按与 video-eval 方案的接近度从高到低排列。

### 1. AIGVE_Tool（IFM Lab）— 最接近，但很早期

- 仓库：https://github.com/ShaneXiangH/AIGVE_Tool
- Star：14
- License：MIT

定位是"AI Generated Video Evaluation Toolkit"，有一个"Metrics Zoo"，集成了 FID、FVD、CLIPSim、VBench、VideoScore 等十几个评估指标，按类别分组（分布对比类、视觉语言相似度类、多维度评估类）。支持自定义指标和模块化 DataLoader，设计上声称可扩展。

**不足**：非常早期（14 star）。没有 Docker、没有设备适配（Mac MPS → CUDA 切换）、没有业务维度（合规、商品还原、CTA、营销逻辑）、没有 VLM-as-Judge 带证据溯源的结构化输出。更像是一个学术指标聚合工具，不是面向业务落地的评估系统。

### 2. VBench（Vchitect, CVPR 2024）— 最主流，但是学术基准

- 仓库：https://github.com/Vchitect/VBench
- Star：1.7k
- License：见仓库

目前 AIGC 视频评估领域 star 最高的项目，pip 可安装。16 个维度覆盖主体一致性、时序闪烁、动态度、美学质量等。VBench-2.0 加了物理合理性、常识推理等 18 个维度。支持 `--mode=custom_input` 传入自己的视频跑已有指标。

**不足**：本质是学术 benchmark，不是可部署的评估系统。没有 Docker、没有插件架构、不支持新增维度、没有设备适配层。评估维度全是"技术画质"类，完全没有商业/业务维度。

### 3. VMAF（Netflix）— 传统视频质量的天花板，但不碰 AIGC

- 仓库：https://github.com/Netflix/vmaf
- Star：5.5k
- License：BSD+Patent

Netflix 的感知视频质量评估算法，支持 PSNR/SSIM/MS-SSIM 等，有 C 库 + Python 库 + FFmpeg 滤镜 + Docker。支持训练自定义 VMAF 模型。

**不足**：评估的是压缩/传输质量（块效应、模糊、抖动），完全不碰 AIGC 瑕疵、多模态理解、商业意图。它的"可扩展"是指加新的传统质量指标，不是加语义/业务维度。

### 4. CenseoQoE（Tencent）— 传统 QoE，不碰 AIGC

- 仓库：https://github.com/Tencent/CenseoQoE
- Star：240
- License：MIT（GitHub 检测为 Other）

做 UGC/PGC/游戏视频的主观质量评估。有训练框架 + C++ SDK 部署。

**不足**：没有 AIGC 评估、没有插件架构、没有 Docker。

### 5. 其他

| 项目 | 类型 | 说明 |
|------|------|------|
| EvalCrafter | 纯 benchmark | 不可安装不可扩展 |
| VideoScore / VIEScore | 学术论文 | 只发布数据集和论文，不是可用工具 |
| TLive-Omni（淘宝直播） | 理解模型 | 是视频理解模型，不是评估框架 |

## 二、竞品对比矩阵

| 能力 | video-eval | AIGVE_Tool | VBench | VMAF | CenseoQoE |
|------|-----------|------------|--------|------|-----------|
| AIGC 视频评估 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 插件化架构 | ✅ | 部分 | ❌ | 部分 | ❌ |
| Docker 部署 | ✅ | ❌ | ❌ | ✅ | ❌ |
| 设备适配（Mac→GPU） | ✅ | ❌ | ❌ | ❌ | ❌ |
| 商业维度（合规/卖点/CTA/营销逻辑） | ✅ | ❌ | ❌ | ❌ | ❌ |
| VLM-as-Judge + 证据溯源 | ✅ | ❌ | 部分 | ❌ | ❌ |
| 商品还原度 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 按视频类型差异化评估 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 社区插件贡献机制 | ✅ | ❌ | ❌ | ❌ | ❌ |

## 三、video-eval 的差异化定位

看完这些项目，video-eval 的差异化可以归纳成三句话：

### 1. 别人评"技术质量"，video-eval 评"商业效果"

VBench/VMAF/AIGVE 评的是画面一致性、时序闪烁、美学质量——这些 video-eval 也会评，但只是基础维度。独特价值是：商品卖点有没有覆盖、前 3 秒钩子够不够强、CTA 有没有、营销逻辑完不完整、合规有没有踩线。这些维度 GitHub 上一个项目都没有。

### 2. 别人是"学术工具"，video-eval 是"可部署系统"

VBench 是给论文用的 benchmark，AIGVE 是指标聚合器。video-eval 做的是 Docker 部署、Mac 调试→GPU 部署、CLI 可用、配置驱动、结构化 JSON 输出可直接接入业务管线的产品。这种"工程化"是开源社区最缺的。

### 3. 别人是"封闭指标集"，video-eval 是"插件生态"

VBench 的 16 个维度写死在代码里，不能加。AIGVE 虽然声称可扩展但没有插件发现机制。video-eval 要的是 ESLint 模式——框架管流程和接口，社区贡献维度插件，通过 entry_points 自动发现。一旦有 3-5 个社区插件，护城河就形成了。

## 四、策略建议

1. **集成而非重复造轮子**：VBench 的技术画质模块可以直接作为 video-eval 的一个内置插件接入，不用重新实现。AIGVE 的某些指标也可以作为可选插件。框架管流程，别人的指标做插件。

2. **第一批社区插件从内部能力输出**：团队自研的 AIGC 瑕疵检测器、商品还原度模型可以包装成插件开源，既是对社区的贡献，也是项目的独特壁垒。

3. **品牌定位跟 VBench/AIGVE 区分清楚**：README 开头就写清楚——这不是又一个 AIGC 视频质量 benchmark，而是一个面向业务落地的、可插拔的视频评估系统。别人评"视频画质好不好"，video-eval 评"这个视频能不能用来带货"。
