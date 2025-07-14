import httpx
import asyncio

async def test_search():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:52059/blog/similar",
            json={"query": "veener, club-prime, laminate"}
        )
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

async def test_recommend_product():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:52059/blog/recommend-product",
            json={
                "context": "I need a durable material for kitchen cabinets that can withstand moisture."
            }
        )
        print("Status Code (recommend):", response.status_code)
        data = response.json()
        print("Response JSON (recommend):", data)
        assert isinstance(data["recommended_products"], list)

if __name__ == "__main__":
    asyncio.run(test_search())
    asyncio.run(test_recommend_product())


