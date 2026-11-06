# 上传 GitHub

## 网页方式

1. 打开现有仓库 `https://github.com/AR321109/MagCal`。
2. 解压交付 ZIP。
3. 在仓库页面选择 **Add file -> Upload files**。
4. 上传 ZIP 内的全部文件和目录，而不是上传 ZIP 本身；让本包中的 README 和 LICENSE 覆盖仓库中的简版文件。
5. 提交说明填写 `Add reproducible MagCal implementation`。
6. 上传后确认仓库首页直接显示 `run.py`、`magcal/`、`configs/`、`scripts/` 和 `tests/`。

## Git 命令方式

```bash
git clone https://github.com/AR321109/MagCal.git
cd MagCal
# 将交付包中的全部文件和目录复制到这里，然后执行：
git add -A
git commit -m "Add reproducible MagCal implementation"
git push origin main
```

上传后在一个新的空环境中重新执行安装、测试和 smoke 命令。论文摘要可写：

```latex
The source code is available at
\url{https://github.com/AR321109/MagCal}.
```

如果仍处于双盲评审阶段，不要使用暴露姓名、手机号或个人提交历史的公开账号链接。
