#!/bin/bash
set -e

chmod +x shells/assistant_rd.sh shells/assistant_augment.sh shells/student_rd.sh

echo "Executing assistant_rd.sh..."
shells/assistant_rd.sh

echo "Executing assistant_augment.sh..."
shells/assistant_augment.sh

echo "Executing student_rd.sh..."
shells/student_rd.sh

echo "All scripts executed successfully." 