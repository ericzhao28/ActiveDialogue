echo "Test partial DST env"
python3 -m tests.test_partial_dst_env
echo "Test partial entropy"
python3 -m tests.test_partial_entropy
echo "Test partial bald"
python3 -m tests.test_partial_bald
echo "Test entropy"
python3 -m tests.test_entropy
echo "Test BALD"
python3 -m tests.test_bald
echo "Test DST env"
python3 -m tests.test_dst_env
echo "Test datasets"
python3 -m tests.test_datasets
echo "Test models"
python3 -m tests.test_models
