"""
Script to clone the top open-source repositories from the Hugging Face GitHub organization.

This script fetches repositories from the Hugging Face GitHub organization, sorts them by their
number of "stars" (a GitHub metric indicating how many users have bookmarked or favorited a repository,
often used as a proxy for popularity or usefulness), and clones the top N repositories locally.

Courtesy: Sayak Paul and Chansung Park.
"""

import os
from dotenv import load_dotenv
import subprocess
from multiprocessing import Pool
from github import Github

# Load environment variables from a .env file if present
load_dotenv()

# Name of the GitHub organization to fetch repositories from
ORG = "huggingface"

# Directory where the repositories will be cloned locally
MIRROR_DIRECTORY = "hf_public_repos"

# Number of top repositories (by stars) to clone
TOK_K = 15


def get_repos(username, access_token=None, include_fork=False):
    """
    Fetches repositories for a particular GitHub user or organization.

    Courtesy: Chansung Park.
    Args:
        username (str): GitHub username or organization name.
        access_token (str, optional): GitHub access token for authentication.
        include_fork (bool, optional): Whether to include forked repositories.

    Returns:
        list of tuples: Each tuple contains (repo_name, stargazers_count).
    """
    g = Github(access_token)
    user = g.get_user(username)

    results = []
    # Iterate through all repositories for the user/org
    for repo in user.get_repos():
        if repo.fork is False:
            # Only include original repositories by default
            results.append((repo.name, repo.stargazers_count))
        else:
            if include_fork is True:
                # Optionally include forks if specified
                results.append((repo.name, repo.stargazers_count))
    print(results)
    return results


def sort_repos_by_stars(repos):
    """
    Sorts a list of repositories by their stargazer count in descending order.

    Args:
        repos (list of tuples): Each tuple is (repo_name, stargazers_count).

    Returns:
        list of tuples: Sorted list of repositories.
    """
    return sorted(repos, key=lambda x: x[1], reverse=True)


def mirror_repository(repository):
    """
    Clones a single repository from GitHub to the local mirror directory.

    Args:
        repository (str): Name of the repository to clone.
    """
    repository_url = f"https://github.com/{ORG}/{repository}.git"
    repository_path = os.path.join(MIRROR_DIRECTORY, repository)

    # Use subprocess to run the git clone command
    subprocess.run(["git", "clone", repository_url, repository_path])


def mirror_repositories():
    """
    Main function to fetch, sort, and clone the top repositories from the organization.
    """
    # Create the mirror directory if it doesn't exist
    if not os.path.exists(MIRROR_DIRECTORY):
        os.makedirs(MIRROR_DIRECTORY)

    # Ensure the GitHub access token is set in the environment
    if not os.environ["GH_ACCESS_TOKEN"]:
        raise ValueError("You must set `GH_ACCESS_TOKEN` as an env variable.")

    # Fetch all repositories for the organization
    repositories = get_repos(ORG, os.environ["GH_ACCESS_TOKEN"])
    # Sort repositories by number of stars (descending)
    sorted_repos = sort_repos_by_stars(repositories)
    # Select the top K repositories
    selected_repos = [x[0] for x in sorted_repos[:TOK_K]]

    print(f"Total repositories found: {len(selected_repos)}.")
    print(selected_repos)
    # Clone repositories in parallel using multiprocessing
    print("Cloning repositories.")
    with Pool() as pool:
        pool.map(mirror_repository, selected_repos)


if __name__ == "__main__":
    # Entry point: start the mirroring process
    mirror_repositories()
