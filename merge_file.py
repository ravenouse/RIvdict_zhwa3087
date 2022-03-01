import json
import argparse
import pathlib

def get_parser(
    parser=argparse.ArgumentParser(
        description="run to merge files"
    ),
):
    parser.add_argument(
        "--file1", type=pathlib.Path, help="path to the file1"
    )
    parser.add_argument("--file2", type=pathlib.Path, help="path to the file2")
    parser.add_argument("--file3", type=pathlib.Path, help="path to the file3")
    parser.add_argument("--file4", type=pathlib.Path, help="path to the file4")
    parser.add_argument("--file5", type=pathlib.Path, help="path to the file5")
    parser.add_argument(
        "--save_dir",
        # type=pathlib.Path,
        default=pathlib.Path('data/all.train.json'),
        help="where to save merged file",
    )
    return parser

def merge_jsonfiles(args):
    file1=args.file1
    file2=args.file2
    file3=args.file3
    file4=args.file4
    file5=args.file5
    save_dir=args.save_dir
    result = list()
    with open(file1,'r') as object:
        print(len(json.load(object)))
        # result.extend(json.load(object))
    with open(file2,'r') as object:
        print(len(json.load(object)))
        # result.extend(json.load(object))
    with open(file3,'r') as object:
        print(len(json.load(object)))
        # result.extend(json.load(object))
    with open(file4,'r') as object:
        print(len(json.load(object)))
        # result.extend(json.load(object))
    with open(file5,'r') as object:
        print(len(json.load(object)))
        # result.extend(json.load(object))
    # with open(save_dir,'w') as output_file:
    #     json.dump(result,output_file)

if __name__ == "__main__":
  args = get_parser().parse_args()
  # print(args)
  merge_jsonfiles(args)