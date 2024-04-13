from error_finders.error_finder_factory import create_error_finder


def main():
    error_finder = create_error_finder()
    sentence = "After went to the store, she buyed some apples and oranges, but forgot to brings her wallet so she couldn't pays for them."
    error_finder.find_error(sentence)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
