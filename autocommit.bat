@echo off

rem 设置变量
set repoPath=D:\Projects\IDEA_Projects\datastructure\datastructure
set commitMessage=Automatic commit by script

rem 导航到代码仓库目录
cd /d %repoPath%

rem 添加所有修改到暂存区
git add .

rem 提交代码
git commit -m "%commitMessage%"

rem 推送到远程仓库
git push
