import os

def main() -> str:
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'department_skills.csv':
                return os.path.join(root, file)
            
if __name__ == "__main__":
    print(main())