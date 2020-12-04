# How to install gambit on GreatLakes?
## Installing Gambit 16.0.1

Download: https://sourceforge.net/projects/gambit/files/gambit15/15.1.1

Unzip the package as `~/gambit-16.0.1`

```
cd ~/gambit-16.0.1
autoreconf -f -i
bash ./configure --prefix=/home/wangyzh/.local
make
make install
```

* Compilation with clang 11.

To test Gambit install:

```
gambit-lcp -h
```