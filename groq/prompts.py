#!/usr/bin/env python3

"""

Groq API Wrapper

A simple wrapper to call Groq API with system and user prompts from files.

"""


import argparse

import sys

from pathlib import Path

from groq import Groq


def read_file(file_path):

    """Read content from a file."""

    try:

        with open(file_path, 'r', encoding='utf-8') as f:

            return f.read().strip()

    except FileNotFoundError:

        print(f"Error: File '{file_path}' not found.", file=sys.stderr)

        sys.exit(1)

    except Exception as e:

        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)

        sys.exit(1)


def call_groq_api(system_prompt, user_prompt):

    """Call Groq API with the provided prompts."""

    try:

        client = Groq()

        

        # Build messages array

        messages = []

        if system_prompt:

            messages.append({

                "role": "system",

                "content": system_prompt

            })

        

        messages.append({

            "role": "user",

            "content": user_prompt

        })

        

        completion = client.chat.completions.create(

            model="deepseek-r1-distill-llama-70b",

            messages=messages,

            temperature=0.08,

            max_completion_tokens=4096,

            top_p=0.95,

            stream=True,

            stop=None,

        )

        

        for chunk in completion:

            print(chunk.choices[0].delta.content or "", end="")

        

        print()  # Add newline at the end

        

    except Exception as e:

        print(f"Error calling Groq API: {e}", file=sys.stderr)

        sys.exit(1)


def main():

    parser = argparse.ArgumentParser(

        description="Call Groq API with system and user prompts from files.",

        formatter_class=argparse.RawDescriptionHelpFormatter,

        epilog="""

Examples:

  python groq_wrapper.py -u user_prompt.txt

  python groq_wrapper.py -s system_prompt.txt -u user_prompt.txt

  python groq_wrapper.py --system sys.txt --user usr.txt

        """

    )

    

    parser.add_argument(

        '-s', '--system',

        type=str,

        help='Path to system prompt file (optional)'

    )

    

    parser.add_argument(

        '-u', '--user',

        type=str,

        required=True,

        help='Path to user prompt file (required)'

    )

    

    args = parser.parse_args()

    

    # Read system prompt if provided

    system_prompt = ""

    if args.system:

        system_prompt = read_file(args.system)

    

    # Read user prompt

    user_prompt = read_file(args.user)

    

    if not user_prompt:

        print("Error: User prompt is empty.", file=sys.stderr)

        sys.exit(1)

    

    # Call the API

    call_groq_api(system_prompt, user_prompt)


if __name__ == "__main__":

    main()