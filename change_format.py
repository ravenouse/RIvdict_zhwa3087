import json
import argparse
import pathlib

def get_parser(parser=argparse.ArgumentParser(description="change the format of the prediction")):
    parser.add_argument(
        "--original",
        type=pathlib.Path,
        help="path to the prediction to change",
        default= 'references/en.test.revdict.complete.json'
    )
    parser.add_argument(
        "--new",
        type=pathlib.Path,
        help="where to put and rename the new file",
        default=pathlib.Path("references/en.test.revdict.complete.json"),
    )
    return parser

def change_format(args):
    with open(args.original) as f:
        data = json.load(f)


    for dict in data:
        id = dict["id"]
        id = id.split(".")
        id[1] = 'revdict'
        id = ".".join(id)
        dict["id"] = id


    data = json.dumps(data)

    with open(args.new, 'w') as x:
        x.write(data)
    print("format has been changed.")

if __name__ == "__main__":
    args = get_parser().parse_args()
    change_format(args)