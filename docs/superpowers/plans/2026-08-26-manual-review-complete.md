# Manual Review Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ATEC App 中持久化完整场次的人工 Review 完成状态，并单独分组显示。

**Architecture:** 在 `gui_state.py` 中集中实现标记文件读写、有效性检查和场次分组，GUI只调用这些接口。标记通过导出报告的路径和纳秒修改时间与传播结果绑定，任何重新传播或重新导出都会使旧标记失效。

**Tech Stack:** Python 3、PyQt5、JSON、现有无框架断言测试。

---

### Task 1: Review完成状态模型

**Files:**
- Modify: `atec_pipeline/gui_state.py`
- Test: `tests/test_gui_state.py`

- [ ] **Step 1: 写失败测试**

测试 `mark_scene_human_review_complete()` 创建元数据、`clear_scene_human_review_complete()` 删除元数据、`load_scene_human_review_completion()` 在报告mtime变化后返回无效。

- [ ] **Step 2: 验证测试失败**

Run: `/usr/bin/python3 tests/test_gui_state.py`
Expected: FAIL，导入的新接口不存在。

- [ ] **Step 3: 最小实现**

新增：

```python
MANUAL_REVIEW_COMPLETE_FILENAME = "manual_review_complete.json"
def mark_scene_human_review_complete(project_root: Path, scene_dir: Path) -> Path:
    """Persist completion bound to the current export report."""

def clear_scene_human_review_complete(scene_dir: Path) -> bool:
    """Remove the marker and return whether it existed."""

def load_scene_human_review_completion(
    project_root: Path, scene_dir: Path
) -> HumanReviewCompletion:
    """Return a valid/invalid completion with an explanatory reason."""
```

将 `SceneWorkflowState` 增加 `human_review_complete`，有效时把完成关键帧的场次分到 `human_reviewed`。

- [ ] **Step 4: 验证测试通过**

Run: `/usr/bin/python3 tests/test_gui_state.py`
Expected: PASS。

### Task 2: App按钮和第三分组

**Files:**
- Modify: `atec_pipeline/gui_app.py`
- Test: `tests/test_gui_app.py`

- [ ] **Step 1: 写失败GUI测试**

断言列表包含“已经过人工 Review”分组；标记按钮写入文件并刷新分组；取消按钮删除标记并恢复分组。

- [ ] **Step 2: 验证GUI测试失败**

Run: `QT_QPA_PLATFORM=offscreen /usr/bin/python3 tests/test_gui_app.py`
Expected: FAIL，第三分组和按钮尚不存在。

- [ ] **Step 3: 最小GUI实现**

在Review区域加入两个按钮，调用 `gui_state.py` 接口；刷新时展示：

```python
groups = (
    ("needs_manual", "未标记完成"),
    ("keyframes_complete", "已标记完成"),
    ("human_reviewed", "已经过人工 Review"),
)
```

任务执行期间禁用按钮，没有有效导出报告时禁止完成标记。

- [ ] **Step 4: 验证GUI测试通过**

Run: `QT_QPA_PLATFORM=offscreen /usr/bin/python3 tests/test_gui_app.py`
Expected: PASS。

### Task 3: 回归验证

**Files:**
- Modify: `docs/zh-CN/APP使用手册.md`（若存在，否则更新当前App手册）

- [ ] **Step 1: 更新使用说明**

说明人工 Review 完成标记、取消和自动失效行为。

- [ ] **Step 2: 运行回归**

```bash
/usr/bin/python3 tests/test_gui_state.py
QT_QPA_PLATFORM=offscreen /usr/bin/python3 tests/test_gui_app.py
python3 -m py_compile atec_pipeline/*.py tools/*.py
git diff --check
./scripts/atec-pipeline smoke-test
```

Expected: 全部通过。

- [ ] **Step 3: 检查Git差异**

确认没有改动 RGB-D、Mask、数据集或训练权重。
