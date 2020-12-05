import pandas
import numpy as np

from pandas import DataFrame


def parse_file(file_name: str) -> DataFrame:
    data: DataFrame = pandas.read_csv(file_name, header=0)

    # Fields to drop: year, artists, id?, name?, release date?
    # Fields to one-hot encode: key, year/release_date?
    # Fields to scale: durability_ms, loudness?, tempo, year?
    # I guess try to predict the artist? year/release date would probably be important then
    return data


def main():
    data = parse_file('data.csv')
    print(data)
    return

main()

