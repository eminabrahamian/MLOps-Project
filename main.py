"a simple example on how to use argparse in a script"
import argparse

def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser(description='Greet the user by name.')
    parser.add_argument('name', help='Name of the person to greet')
    args = parser.parse_args()

    try:
        print(f'Hello, {args.name}!')
    except Exception as e:
        print(f'Error: {e}')
        raise

if __name__ == '__main__':
    main()