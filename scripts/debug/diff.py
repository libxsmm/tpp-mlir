#!/usr/bin/env python3

import os
import argparse
import re
import subprocess

def syntax():
    print("Use -h for options\n")
    exit(1)

class File():
    def __init__(self, path: str, ext: str, counter: int):
        self.path = path
        self.ext = ext
        self.counter = counter
        self.pattern = f"{self.path}/%03d.{self.ext}"
        self.lines = list()
        self.pass_pattern = "\\([\\w-]+\\)"
        self.pass_name = ""

    def getFileName(self):
        return self.pattern % self.counter
    
    def fileExists(self) -> bool:
        return os.path.isfile(self.getFileName())

    def read(self):
        with open(self.getFileName(), "r") as fp:
            for line in fp:
                if line is None or line == "":
                    continue
                if line.startswith("//"):
                    m = re.search(self.pass_pattern, line)
                    if m:
                        self.pass_name = m.group(0)
                    continue
                self.lines.append(line)

    def getPassName(self) -> str:
        return self.pass_name

    def isSamePass(self, other: "File") -> bool:
        return self.pass_name == other.pass_name

    def __eq__(self, other: "File") -> bool:
        return self.lines == other.lines

    def __lt__(self, other: "File") -> bool:
        return self.getFileName() < other.getFileName()

    @staticmethod
    def get(path: str, ext: str, counter: int) -> "File":
        f = File(path, ext, counter)
        if not f.fileExists():
            return None
        f.read()
        return f

class Directory():
    def __init__(self, path: str, ext: str):
        self.path = path
        self.ext = ext
        self.files = list()
        self.pass_list = list()
    
    def exists(self):
        return os.path.isdir(self.path)
    
    def getAllFiles(self):
        counter = 0
        while True:
            counter += 1
            f = File.get(self.path, self.ext, counter)
            if f is None:
                break
            self.files.append(f)

    def isEmpty(self):
        self.getAllFiles()
        return len(self.files) == 0
    
    def numFiles(self) -> int:
        return len(self.files)

    def getFile(self, counter: int) -> "File":
        if (counter < 0 or counter >= len(self.files)):
            return None
        return self.files[counter]

    def getPass(self, counter: int) -> str:
        if (counter < 0 or counter >= len(self.files)):
            return None
        return self.files[counter].getPassName()

    def isSamePass(self, counter: int, other: "File"):
        if (counter < 0 or counter >= len(self.files)):
            return None
        return self.files[counter].getPassCode() == other.getPassCode()

    @staticmethod
    def get(path: str, ext: str) -> "Directory":
        dir = Directory(path, ext)
        if not dir.exists():
            print(f"{path} is not a directory\n")
            syntax()
        if dir.isEmpty():
            print(f"No files found in {path} with extension {ext}\n")
            syntax()
        return dir

def compareTwoFiles(file1: File, file2: File, diffTool: str):
    if file1 == file2:
        print(f"Files {file1.getFileName()} and {file2.getFileName()} are identical, skipping...\n")
        return

    cmd = [diffTool, file1.getFileName(), file2.getFileName()]
    #print(" ".join(cmd))
    print("Press ENTER to continue or CTRL+C to cancel...\n")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCtrl+C was pressed. Aborting...\n")
        exit(0)

    subprocess.run(cmd)

def compareDifferentPipeline(path: str, baseline: str, ext: str, diffTool: str, read: int, skip: int):
    baseDir = Directory.get(baseline, ext)
    currDir = Directory.get(path, ext)
    counter = skip
    last = baseDir.numFiles() if read < 0 else (skip + read)

    while True:
        baseFile = baseDir.getFile(counter)
        if baseFile is None:
            print(f"No more files\n")
            break
        passName = baseFile.getPassName()
        if passName in ("(canonicalize)", "(cse)", "(cleanup)"):
            print(f"Skipping pass {passName}...\n")
            counter += 1
            continue
        print(f"Looking for same pass as {baseFile.getPassName()}...\n")
        innerCounter = counter
        while True:
            currFile = currDir.getFile(innerCounter)
            if currFile is None:
                print(f"Cannot find same pass as {baseFile.getPassName()}, skipping...\n")
                break
            if not baseFile.isSamePass(currFile):
                innerCounter += 1
                continue

            compareTwoFiles(baseFile, currFile, diffTool)
            innerCounter += 1
            break
        counter += 1
        if (counter >= last):
            print(f"Reached read limit of {read} files\n")
            break


def compareSamePipeline(path: str, ext: str, diffTool: str, read: int, skip: int):
    dir = Directory.get(path, ext)
    counter = skip
    last = dir.numFiles() if read < 0 else (skip + read)
    curr = dir.getFile(counter)
    if curr is None:
        print(f"No numbered files found in {path} with extension {ext}\n")
        syntax()

    while True:
        counter += 1
        next = dir.getFile(counter)
        if next is None:
            print(f"No more files\n")
            break
        compareTwoFiles(curr, next, diffTool)
        curr = next
        if (counter >= last):
            print(f"Reached read limit of {read} files\n")
            break

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", help="directory with files to compare")
    argParser.add_argument("-d", "--diff", help="diff tool to use")
    argParser.add_argument("-b", "--base", help="baseline directory (to compare with --path)")
    argParser.add_argument("-s", "--skip", type=int, help="number of initial passes to skip", default=1)
    argParser.add_argument("-r", "--read", type=int, help="number of passes to read", default=-1)
    argParser.add_argument("ext", nargs="?", help="file extension to look for", default="mlir")
    args = argParser.parse_args()
    if args.ext is None:
        syntax()

    if args.path is None:
        args.path = "."

    if not os.path.isdir(args.path):
        print(f"{args.path} not a directory\n")
        syntax()

    if args.diff is None:
        args.diff = "diff"

    if args.base is None:
        compareSamePipeline(args.path, args.ext, args.diff, args.read, args.skip)
    else:
        compareDifferentPipeline(args.path, args.base, args.ext, args.diff, args.read, args.skip)