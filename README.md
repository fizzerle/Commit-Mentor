# Semi-Automatic-Commit-Generation

To run the flask Server type in the CMD:

    set FLASK_APP=SemiAutomaticCommitSystem.py
    set FLASK_ENV=development
    flask run

To install the hook for automatic execution of the system:
    
    pre-commit install --hook-type prepare-commit-msg
