#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка текстового пересечения с более ранними редакциями работы.

Полноценный отчёт «Антиплагиата» этим не заменяется: сервис сравнивает текст с
закрытыми коллекциями (РГБ, eLIBRARY, вузовские хранилища), доступа к которым
нет. Здесь решается более узкая, но практически важная задача — оценить
САМОповтор: если ранняя редакция была где-то выложена, совпадающие фрагменты
попадут в отчёт как заимствование.

Метод: шинглы по 8 слов (нормализованные), доля шинглов статьи, встречающихся
в каждом из сравниваемых файлов.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNLOADS = HERE.parent.parent          # ...\Downloads
K = 8                                   # длина шингла в словах


def plain(path):
    """Грубое снятие LaTeX-разметки."""
    t = path.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"(?m)^\s*%.*$", " ", t)                 # комментарии
    t = re.sub(r"\\begin\{(equation|align|tabular|table|figure)\*?\}.*?"
               r"\\end\{\1\*?\}", " ", t, flags=re.S)   # формулы и таблицы
    t = re.sub(r"\$[^$]*\$", " ", t)                    # инлайн-математика
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)  # команды
    t = re.sub(r"[{}~\\]", " ", t)
    t = t.lower()
    return re.findall(r"[а-яёa-z]+", t)


def shingles(words, k=K):
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def main():
    me = HERE / "paper_ru.tex"
    my_words = plain(me)
    my_sh = shingles(my_words)
    print("paper_ru.tex: %d слов, %d шинглов по %d слов\n"
          % (len(my_words), len(my_sh), K))

    cands = sorted(p for p in DOWNLOADS.glob("*.tex") if p.resolve() != me.resolve())
    if not cands:
        print("ранних редакций рядом не найдено")
        return 0

    print("%-38s %8s %9s %s" % ("файл", "слов", "пересеч.", "доля шинглов статьи"))
    print("-" * 78)
    worst = 0.0
    for c in cands:
        w = plain(c)
        sh = shingles(w)
        inter = my_sh & sh
        frac = len(inter) / max(len(my_sh), 1)
        worst = max(worst, frac)
        print("%-38s %8d %9d %17.1f%%" % (c.name[:38], len(w), len(inter), 100 * frac))

    print()
    print("Наибольшее самоперекрытие: %.1f%%" % (100 * worst))
    print()
    print("Как читать: это НЕ показатель оригинальности. Совпадение с")
    print("собственными неопубликованными черновиками безвредно; опасно оно")
    print("только если черновик был выложен в открытый доступ (arXiv, репозиторий,")
    print("препринт-сервер). Проверьте, публиковались ли перечисленные файлы.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
