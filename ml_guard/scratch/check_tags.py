
import sys

def check_tags(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    stack = []
    for i, line in enumerate(lines):
        # Very simple tag extractor
        pos = 0
        while True:
            start = line.find('<', pos)
            if start == -1: break
            
            end = line.find('>', start)
            if end == -1: break
            
            tag_content = line[start+1:end].strip()
            pos = end + 1
            
            if tag_content.startswith('!--'): continue # comment
            if tag_content.endswith('/'): continue # self-closing
            
            if tag_content.startswith('/'):
                tag_name = tag_content[1:].split()[0]
                if not stack:
                    print(f"Error: Closing tag </{tag_name}> at line {i+1} has no matching opening tag")
                else:
                    last_tag = stack.pop()
                    if last_tag != tag_name:
                        print(f"Error: Mismatched tag at line {i+1}. Expected </{last_tag}>, got </{tag_name}>")
            else:
                # Opening tag
                tag_name = tag_content.split()[0]
                # Filter out React components (uppercase) and non-div tags for now to focus
                if tag_name == 'div':
                    stack.append(tag_name)
    
    if stack:
        print(f"Error: Unclosed tags: {stack}")
    else:
        print("All div tags are balanced!")

if __name__ == "__main__":
    check_tags(sys.argv[1])
